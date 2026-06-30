"""Red neuronal bayesiana de última capa con MLX, contrastada con NUTS de PyMC.

Pipeline del laboratorio:

1. Datos sintéticos heterocedásticos (ruido log-normal cuya dispersión crece con x).
2. Un MLP funcional entrenado por SGD (momentum) con verosimilitud t-Student
   (cabeza heterocedástica: salida = mu, log-varianza y nu de la t). Las colas
   pesadas de la t la hacen robusta a los valores atípicos.
3. Bayesiano de **última capa**: congelamos el cuerpo en su valor MAP como
   extractor de características phi(x) y muestreamos SOLO la última capa lineal.
   Es barato y captura la mayor parte de la incertidumbre epistémica.
   - con ``blmx.sample_nuts_chains`` (NUTS sobre MLX), y
   - con el NUTS de **PyMC** sobre el modelo equivalente (mismas características
     congeladas como datos, misma previa N(0,1), misma verosimilitud t-Student).
4. Comparamos las métricas predictivas de ambos muestreadores (deben coincidir)
   frente al MLP determinista, y dibujamos las curvas.

Todo el backend numérico es MLX (Apple Silicon); no se usa JAX/distrax/TFP. La t
de Student se implementa a mano (su constante de normalización necesita lgamma,
que aproximamos con Lanczos, diferenciable y válido para nu por punto).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from blmx import sample_nuts_chains

RANDOM_STATE = 42   # semilla única para todo el laboratorio (reproducibilidad)
PRIOR_SD = 1.0      # previa N(0, PRIOR_SD) sobre los pesos bayesianos


# ---------------------------------------------------------------------------
# lgamma diferenciable (aproximación de Lanczos): la constante de normalización
# de la t de Student depende de nu (parámetro por punto), así que necesitamos
# log Gamma trazable por MLX.
# ---------------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
]
_LOG_SQRT_2PI = float(0.5 * np.log(2 * np.pi))


def lgamma(x):
    """log Gamma(x) para x > 0 (Lanczos), elementwise y diferenciable en MLX."""
    x = x - 1.0
    a = mx.array(_LANCZOS_C[0])
    t = x + _LANCZOS_G + 0.5
    for i in range(1, _LANCZOS_G + 2):
        a = a + _LANCZOS_C[i] / (x + i)
    return (x + 0.5) * mx.log(t) - t + mx.log(a) + _LOG_SQRT_2PI


def tstudent_logpdf(y, mu, sigma, nu):
    """log-densidad de una t de Student (location-scale) por punto, en MLX."""
    z = (y - mu) / sigma
    return (
        lgamma(0.5 * (nu + 1.0)) - lgamma(0.5 * nu)
        - 0.5 * mx.log(nu * np.pi) - mx.log(sigma)
        - 0.5 * (nu + 1.0) * mx.log1p(z * z / nu)
    )


def f(x):
    """Función verdadera (media condicional) sin ruido."""
    return x * np.sin(x)


def train_test_split(X, y, test_size=0.3, random_state=0):
    """División entrenamiento/prueba sin dependencias externas (sklearn)."""
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(X))
    n_test = int(round(test_size * len(X)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------------------------------------------------------------------------
# MLP funcional (lista de pares (W, b)); compatible con el bucle SGD de abajo.
# ---------------------------------------------------------------------------
def init_mlp(key, sizes):
    """Inicializa los parámetros de un MLP con inicialización tipo He."""
    params = []
    keys = mx.random.split(key, len(sizes) - 1)
    for k, din, dout in zip(keys, sizes[:-1], sizes[1:]):
        W = mx.random.normal((din, dout), key=k) * np.sqrt(2.0 / din)
        b = mx.zeros((dout,))
        params.append([W, b])
    return params


def mlp_forward(params, x):
    """Pasada hacia adelante: capas ReLU ocultas y salida lineal."""
    for W, b in params[:-1]:
        x = nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def mlp_features(body_params, x):
    """Extractor de características: todas las capas menos la última, con ReLU."""
    for W, b in body_params:
        x = nn.relu(x @ W + b)
    return x


def predict_params(out):
    """(mu, sigma, nu) a partir de la salida (N, 3) de la cabeza t-Student."""
    mu = out[:, 0]
    sigma = mx.exp(0.5 * mx.clip(out[:, 1], -7.0, 7.0))  # estabilidad numérica
    nu = 2.0 + nn.softplus(out[:, 2])                    # nu > 2 (varianza finita)
    return mu, sigma, nu


def tstudent_loss(params, x, y):
    """NLL de una t de Student (location-scale) con cabeza heterocedástica."""
    mu, sigma, nu = predict_params(mlp_forward(params, x))
    return -mx.mean(tstudent_logpdf(y, mu, sigma, nu))


def train(loss_fn, params, X, y, n_epochs=4000, lr=1e-2, momentum=0.9, log_every=2000):
    """Entrena `params` minimizando `loss_fn(params, X, y)` con SGD (momentum)."""
    velocity = tree_map(lambda p: mx.zeros_like(p), params)
    loss_and_grad = mx.value_and_grad(loss_fn)
    for epoch in range(1, n_epochs + 1):
        loss, grads = loss_and_grad(params, X, y)
        velocity = tree_map(lambda v, g: momentum * v + g, velocity, grads)
        params = tree_map(lambda p, v: p - lr * v, params, velocity)
        mx.eval(params, velocity)
        if epoch % log_every == 0 or epoch == 1:
            print("  época %5d  |  pérdida = %.4f" % (epoch, loss.item()))
    return params


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------
rng = np.random.RandomState(RANDOM_STATE)
n_samples = 600
X = rng.uniform(0.0, 10.0, size=n_samples).reshape(-1, 1)
expected_y = f(X).ravel()
# Ruido log-normal cuya dispersión aumenta con x (heterocedástico y asimétrico)
sigma_noise = 0.5 + X.ravel() / 10.0
noise = rng.lognormal(mean=0.0, sigma=sigma_noise) - np.exp(sigma_noise**2 / 2.0)
y = expected_y + noise
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)
print("Entrenamiento: %d muestras | Prueba: %d muestras" % (len(X_train), len(X_test)))

x_mean, x_sd = X_train.mean(), X_train.std()
y_mean, y_sd = y_train.mean(), y_train.std()
standardize_X = lambda a: (a - x_mean) / x_sd

# Tensores MLX estandarizados (float32).
Xtr = mx.array(standardize_X(X_train).astype(np.float32))
ytr = mx.array(((y_train - y_mean) / y_sd).astype(np.float32))
Xte = mx.array(standardize_X(X_test).astype(np.float32))
yte = mx.array(((y_test - y_mean) / y_sd).astype(np.float32))

xx = np.linspace(0.0, 10.0, 400).reshape(-1, 1)
Xgrid = mx.array(standardize_X(xx).astype(np.float32))


# ---------------------------------------------------------------------------
# 1) Entrenamiento del MLP determinista (SGD) con verosimilitud t-Student.
# ---------------------------------------------------------------------------
key = mx.random.key(RANDOM_STATE)
key, k_init = mx.random.split(key)
params = init_mlp(k_init, [1, 64, 64, 3])
print("\nEntrenando MLP con pérdida t-Student (SGD)...")
params = train(tstudent_loss, params, Xtr, ytr, n_epochs=8000, lr=1e-2)


# ---------------------------------------------------------------------------
# 2) Bayesiano de última capa: congelamos el cuerpo MAP como phi(x) y
#    muestreamos SOLO la última capa lineal (W_L, b_L) con previa N(0, 1).
#        log p(W_L, b_L | datos) = sum_i log t(y_i | head(phi(x_i))) - 0.5||·||^2
# ---------------------------------------------------------------------------
body_params = params[:-1]            # capas congeladas (valor MAP)
W_last0, b_last0 = params[-1]        # (W_L, b_L) MAP: punto de arranque
H = W_last0.shape[0]                  # dimensión de las características
N_OUT = W_last0.shape[1]
model_ndim_last = H * N_OUT + N_OUT

# Características congeladas (sin gradiente hacia el cuerpo), como datos fijos.
phi_tr = mx.stop_gradient(mlp_features(body_params, Xtr))
phi_te = mx.stop_gradient(mlp_features(body_params, Xte))
phi_grid = mx.stop_gradient(mlp_features(body_params, Xgrid))


def unflatten_last(flat):
    """Vector -> (W_L (H, N_OUT), b_L (N_OUT,))."""
    W = flat[: H * N_OUT].reshape(H, N_OUT)
    b = flat[H * N_OUT:]
    return W, b


def head_from_phi(flat, phi):
    """(mu, sigma, nu) de la última capa lineal sobre características phi."""
    W, b = unflatten_last(flat)
    return predict_params(phi @ W + b)


def log_posterior_last(flat):
    """log-posterior no normalizado sobre la última capa (cuerpo congelado)."""
    mu, sigma, nu = head_from_phi(flat, phi_tr)
    loglik = mx.sum(tstudent_logpdf(ytr, mu, sigma, nu))
    logprior = -0.5 * mx.sum(flat * flat) / (PRIOR_SD ** 2)
    return loglik + logprior


# Contrato del backend MLX: q (vector) -> (logp escalar, dlogp); lo batchea el
# muestreador sobre las cadenas con mx.vmap.
logp_dlogp_last = mx.value_and_grad(log_posterior_last)
flat0 = mx.concatenate([W_last0.reshape(-1), b_last0])

print("\nMuestreando la última capa con NUTS-MLX (%d parámetros)..." % model_ndim_last)
trace_mlx, stats_mlx = sample_nuts_chains(
    logp_dlogp_last, model_ndim_last,
    draws=500, tune=500, chains=4, target_accept=0.9,
    start=flat0, random_seed=RANDOM_STATE,
)
samples_mlx = np.asarray(trace_mlx).reshape(-1, model_ndim_last)
print("  muestras: %d  |  divergencias: %d" % (samples_mlx.shape[0], int(stats_mlx["diverging"].sum())))


# ---------------------------------------------------------------------------
# 2b) Mismo modelo de última capa, muestreado con el NUTS de PyMC (referencia).
# ---------------------------------------------------------------------------
def sample_pymc_last():
    import pymc as pm
    import pytensor.tensor as pt

    phi_np = np.asarray(phi_tr)
    ytr_np = np.asarray(ytr)
    with pm.Model():
        W = pm.Normal("W", 0.0, PRIOR_SD, shape=(H, N_OUT))
        b = pm.Normal("b", 0.0, PRIOR_SD, shape=N_OUT)
        out = pt.dot(phi_np, W) + b
        mu = out[:, 0]
        sigma = pt.exp(0.5 * pt.clip(out[:, 1], -7.0, 7.0))
        nu = 2.0 + pt.softplus(out[:, 2])
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=ytr_np)
        idata = pm.sample(
            draws=500, tune=500, chains=4, cores=1, target_accept=0.9,
            random_seed=RANDOM_STATE, progressbar=False,
        )
    post = idata.posterior
    W_s = np.asarray(post["W"]).reshape(-1, H, N_OUT)
    b_s = np.asarray(post["b"]).reshape(-1, N_OUT)
    flat = np.concatenate([W_s.reshape(W_s.shape[0], -1), b_s], axis=1)
    n_div = int(idata.sample_stats["diverging"].values.sum())
    return flat, n_div


print("\nMuestreando la última capa con NUTS-PyMC (referencia)...")
samples_pm, n_div_pm = sample_pymc_last()
print("  muestras: %d  |  divergencias: %d" % (samples_pm.shape[0], n_div_pm))


# ---------------------------------------------------------------------------
# Métricas predictivas en el conjunto de prueba (unidades reales).
# ---------------------------------------------------------------------------
yte_real = y_test
phi_te_np = np.asarray(phi_te)


def head_np(flat, phi):
    """(mu, sigma, nu) en numpy para un vector de parámetros de última capa."""
    W = flat[: H * N_OUT].reshape(H, N_OUT)
    b = flat[H * N_OUT:]
    out = phi @ W + b
    mu = out[:, 0]
    sigma = np.exp(0.5 * np.clip(out[:, 1], -7.0, 7.0))
    nu = 2.0 + np.log1p(np.exp(out[:, 2]))
    return mu, sigma, nu


def rmse(pred_real):
    return float(np.sqrt(np.mean((pred_real - yte_real) ** 2)))


def coverage95(pred_samples):
    """Cobertura empírica del IC predictivo central del 95% (ideal: 0.95)."""
    lo = np.percentile(pred_samples, 2.5, axis=0)
    hi = np.percentile(pred_samples, 97.5, axis=0)
    return float(np.mean((yte_real >= lo) & (yte_real <= hi)))


def crps(pred_samples):
    """CRPS medio estimado de muestras predictivas (estimador ordenado)."""
    M = pred_samples.shape[0]
    term1 = np.mean(np.abs(pred_samples - yte_real[None, :]), axis=0)
    xs = np.sort(pred_samples, axis=0)
    i = np.arange(1, M + 1)[:, None]
    term2 = np.sum((2.0 * i - M - 1.0) * xs, axis=0) / (M ** 2)
    return float(np.mean(term1 - term2))


def bayes_metrics(samples, rng_seed):
    """RMSE, log-lik predictiva/punto, cobertura 95% y CRPS de un posterior."""
    rng = np.random.RandomState(rng_seed)
    S = samples.shape[0]
    mus, lps, preds = [], [], []
    for flat in samples:
        mu, sigma, nu = head_np(flat, phi_te_np)
        mu_real = mu * y_sd + y_mean
        sigma_real = sigma * y_sd
        mus.append(mu_real)
        lps.append(student_t.logpdf(yte_real, df=nu, loc=mu_real, scale=sigma_real))
        preds.append(mu_real + sigma_real * rng.standard_t(nu))
    mus, lps, preds = np.array(mus), np.array(lps), np.array(preds)
    mean_mu = mus.mean(axis=0)
    # log-verosimilitud predictiva: log((1/S) sum_s p_s(y)) promediada sobre puntos
    from scipy.special import logsumexp
    ll = float(np.mean(logsumexp(lps, axis=0) - np.log(S)))
    return rmse(mean_mu), ll, coverage95(preds), crps(preds)


# MLP determinista (la moda del modelo bayesiano).
mu_d, sigma_d, nu_d = head_np(np.asarray(flat0), phi_te_np)
det_mu = mu_d * y_sd + y_mean
det_rmse = rmse(det_mu)
det_ll = float(np.mean(student_t.logpdf(yte_real, df=nu_d, loc=det_mu, scale=sigma_d * y_sd)))

mlx_rmse, mlx_ll, mlx_cov, mlx_crps = bayes_metrics(samples_mlx, RANDOM_STATE)
pm_rmse, pm_ll, pm_cov, pm_crps = bayes_metrics(samples_pm, RANDOM_STATE + 1)

print("\n=== Comparación en el conjunto de prueba ===")
hdr = "%-34s%9s%13s%10s%9s" % ("Modelo", "RMSE", "log-lik/pto", "cob.95%", "CRPS")
print(hdr)
print("%-34s%9.4f%13.4f%10s%9s" % ("MLP determinista (MAP)", det_rmse, det_ll, "-", "-"))
print("%-34s%9.4f%13.4f%10.3f%9.4f" % ("Bayes última capa (NUTS-MLX)", mlx_rmse, mlx_ll, mlx_cov, mlx_crps))
print("%-34s%9.4f%13.4f%10.3f%9.4f" % ("Bayes última capa (NUTS-PyMC)", pm_rmse, pm_ll, pm_cov, pm_crps))
print("(log-lik mayor es mejor; CRPS menor es mejor; cobertura 95% ideal ≈ 0.95)")
print("Los dos muestreadores bayesianos deben coincidir; ambos calibran mejor que el MAP.")


# ---------------------------------------------------------------------------
# Gráfica: determinista vs bayesiano (NUTS-MLX vs NUTS-PyMC) en la malla.
# ---------------------------------------------------------------------------
def grid_post_mean(samples):
    phi_g = np.asarray(phi_grid)
    mus = np.array([head_np(flat, phi_g)[0] for flat in samples]) * y_sd + y_mean
    return mus.mean(axis=0), np.percentile(mus, 2.5, axis=0), np.percentile(mus, 97.5, axis=0)


det_grid = head_np(np.asarray(flat0), np.asarray(phi_grid))[0] * y_sd + y_mean
mlx_mean, mlx_lo, mlx_hi = grid_post_mean(samples_mlx)
pm_mean, _, _ = grid_post_mean(samples_pm)

plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, s=8, alpha=0.2, color="gray", label="datos")
plt.plot(xx, f(xx), "k--", label="media verdadera")
plt.plot(xx, det_grid, "C1", label="MLP determinista (MAP)")
plt.plot(xx, mlx_mean, "C0", label="Bayes última capa (NUTS-MLX)")
plt.plot(xx, pm_mean, "C2", lw=1, ls=":", label="Bayes última capa (NUTS-PyMC)")
plt.fill_between(xx.ravel(), mlx_lo, mlx_hi, color="C0", alpha=0.2,
                 label="IC 95% (incertidumbre epistémica, MLX)")
plt.legend()
plt.title("Determinista (SGD/MAP) vs Bayesiano de última capa — t-Student")
plt.tight_layout()
plt.savefig("bayesian_neural_net.png", dpi=120)
print("\nGráfica guardada en bayesian_neural_net.png")
