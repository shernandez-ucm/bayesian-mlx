"""Red neuronal bayesiana (última capa vs red completa) con MLX, usando HMC.

Pipeline del laboratorio:

1. Datos sintéticos heterocedásticos (ruido log-normal cuya dispersión crece con x).
2. Un MLP funcional entrenado por SGD (momentum) con verosimilitud t-Student
   (cabeza heterocedástica: salida = mu, log-varianza y nu de la t). Las colas
   pesadas de la t la hacen robusta a los valores atípicos.
3. Bayesiano de **última capa**: congelamos el cuerpo en su valor MAP como
   extractor de características phi(x) y muestreamos SOLO la última capa lineal
   con ``blmx.sample_hmc_chains``. Es barato y captura la mayor parte de la
   incertidumbre epistémica.
4. Bayesiano de **red completa**: muestreamos TODOS los parámetros (cuerpo +
   última capa) con HMC-MLX, capturando la incertidumbre epistémica completa.
5. Comparamos las métricas predictivas y el rendimiento (pasos/seg, grads/paso)
   de ambos enfoques frente al MLP determinista, y dibujamos las curvas.

Todo el backend numérico es MLX (Apple Silicon); no se usa JAX/distrax/TFP. La t
de Student se implementa a mano (su constante de normalización necesita lgamma,
que aproximamos con Lanczos, diferenciable y válido para nu por punto).
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from blmx import sample_hmc_chains

RANDOM_STATE = 42   # semilla única para todo el laboratorio (reproducibilidad)
PRIOR_SD = 1.0      # previa N(0, PRIOR_SD) sobre los pesos bayesianos


class GradientCounter:
    """Contador de evaluaciones de gradiente para medir eficiencia del muestreador."""
    def __init__(self):
        self.count = 0

    def reset(self):
        self.count = 0

    def wrap(self, logp_dlogp_fn):
        """Envuelve una función logp_dlogp para contar llamadas."""
        def counted_logp_dlogp(q):
            self.count += 1
            return logp_dlogp_fn(q)
        return counted_logp_dlogp


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
params = train(tstudent_loss, params, Xtr, ytr, n_epochs=20000, lr=1e-2)


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
logp_dlogp_last_base = mx.value_and_grad(log_posterior_last)
flat0 = mx.concatenate([W_last0.reshape(-1), b_last0])

# Muestrear última capa con HMC-MLX, midiendo rendimiento
print("\nMuestreando la última capa con HMC-MLX (%d parámetros)..." % model_ndim_last)
counter_last = GradientCounter()
logp_dlogp_last = counter_last.wrap(logp_dlogp_last_base)
t0_last = time.time()
trace_mlx, stats_mlx = sample_hmc_chains(
    logp_dlogp_last, model_ndim_last,
    draws=500, tune=500, chains=4, target_accept=0.9,
    start=flat0, random_seed=RANDOM_STATE,
)
time_last = time.time() - t0_last
samples_mlx = np.asarray(trace_mlx).reshape(-1, model_ndim_last)
steps_last = samples_mlx.shape[0]
grads_per_step_last = counter_last.count / max(steps_last, 1) if steps_last > 0 else 0
steps_per_sec_last = steps_last / time_last if time_last > 0 else 0
print("  muestras: %d  |  divergencias: %d" % (steps_last, int(stats_mlx["diverging"].sum())))
print("  rendimiento: %.1f pasos/seg  |  %.2f grads/paso" % (steps_per_sec_last, grads_per_step_last))


# ---------------------------------------------------------------------------
# 2a) Bayesiano de RED COMPLETA: muestreamos TODOS los parámetros (cuerpo + última capa)
#     con previa N(0, PRIOR_SD). Comparamos contra el enfoque de última capa.
# ---------------------------------------------------------------------------
def flatten_params(params_list):
    """Concatena todos los parámetros en un vector."""
    flat_list = []
    for W, b in params_list:
        flat_list.append(W.reshape(-1))
        flat_list.append(b)
    return mx.concatenate(flat_list)


def unflatten_params(flat, sizes):
    """Vector -> lista de pares (W, b) con las formas especificadas."""
    params_out = []
    idx = 0
    for din, dout in zip(sizes[:-1], sizes[1:]):
        W_size = din * dout
        W = flat[idx:idx + W_size].reshape(din, dout)
        b = flat[idx + W_size:idx + W_size + dout]
        params_out.append([W, b])
        idx += W_size + dout
    return params_out


def mlp_forward_full(params_flat, sizes, x):
    """Pasada forward de un MLP desde parámetros aplanados."""
    params = unflatten_params(params_flat, sizes)
    for W, b in params[:-1]:
        x = nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def log_posterior_full(flat):
    """log-posterior no normalizado sobre TODOS los parámetros de la red."""
    sizes = [1, 64, 64, 3]
    out = mlp_forward_full(flat, sizes, Xtr)
    mu, sigma, nu = predict_params(out)
    loglik = mx.sum(tstudent_logpdf(ytr, mu, sigma, nu))
    logprior = -0.5 * mx.sum(flat * flat) / (PRIOR_SD ** 2)
    return loglik + logprior


flat_full0 = flatten_params(params)
model_ndim_full = flat_full0.size

print("\nMuestreando RED COMPLETA con HMC-MLX (%d parámetros)..." % model_ndim_full)
logp_dlogp_full_base = mx.value_and_grad(log_posterior_full)

counter_full = GradientCounter()
logp_dlogp_full = counter_full.wrap(logp_dlogp_full_base)
t0_full = time.time()
trace_mlx_full, stats_mlx_full = sample_hmc_chains(
    logp_dlogp_full, model_ndim_full,
    draws=500, tune=500, chains=4, target_accept=0.9,
    start=flat_full0, random_seed=RANDOM_STATE,
)
time_full = time.time() - t0_full
samples_mlx_full = np.asarray(trace_mlx_full).reshape(-1, model_ndim_full)
steps_full = samples_mlx_full.shape[0]
grads_per_step_full = counter_full.count / max(steps_full, 1) if steps_full > 0 else 0
steps_per_sec_full = steps_full / time_full if time_full > 0 else 0
print("  muestras: %d  |  divergencias: %d" % (steps_full, int(stats_mlx_full["diverging"].sum())))
print("  rendimiento: %.1f pasos/seg  |  %.2f grads/paso" % (steps_per_sec_full, grads_per_step_full))


# ---------------------------------------------------------------------------
# Métricas predictivas en el conjunto de prueba (unidades reales).
# ---------------------------------------------------------------------------
yte_real = y_test
phi_te_np = np.asarray(phi_te)
Xte_np = np.asarray(Xte)


def head_np(flat, phi):
    """(mu, sigma, nu) en numpy para un vector de parámetros de última capa."""
    W = flat[: H * N_OUT].reshape(H, N_OUT)
    b = flat[H * N_OUT:]
    out = phi @ W + b
    mu = out[:, 0]
    sigma = np.exp(0.5 * np.clip(out[:, 1], -7.0, 7.0))
    nu = 2.0 + np.log1p(np.exp(out[:, 2]))
    return mu, sigma, nu


def predict_full_network_np(flat_params, x_np, sizes):
    """Predice (mu, sigma, nu) desde parámetros de red completa en numpy."""
    params = []
    idx = 0
    for din, dout in zip(sizes[:-1], sizes[1:]):
        W_size = din * dout
        W = flat_params[idx:idx + W_size].reshape(din, dout)
        b = flat_params[idx + W_size:idx + W_size + dout]
        params.append([W, b])
        idx += W_size + dout

    # Forward pass
    x = x_np
    for W, b in params[:-1]:
        x = np.maximum(x @ W + b, 0)  # ReLU
    W, b = params[-1]
    out = x @ W + b

    # Parámetros de la t de Student
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

# Métricas para la red completa
sizes_full = [1, 64, 64, 3]
samples_full_np = np.asarray(samples_mlx_full)
mlx_full_rmse, mlx_full_ll, mlx_full_cov, mlx_full_crps = (
    float("nan"), float("nan"), float("nan"), float("nan")
)
if samples_full_np.shape[0] > 0:
    rng = np.random.RandomState(RANDOM_STATE)
    S = samples_full_np.shape[0]
    mus_full, lps_full, preds_full = [], [], []
    for flat in samples_full_np:
        mu, sigma, nu = predict_full_network_np(flat, Xte_np, sizes_full)
        mu_real = mu * y_sd + y_mean
        sigma_real = sigma * y_sd
        mus_full.append(mu_real)
        lps_full.append(student_t.logpdf(yte_real, df=nu, loc=mu_real, scale=sigma_real))
        preds_full.append(mu_real + sigma_real * rng.standard_t(nu))
    mus_full = np.array(mus_full)
    lps_full = np.array(lps_full)
    preds_full = np.array(preds_full)
    from scipy.special import logsumexp
    mean_mu_full = mus_full.mean(axis=0)
    mlx_full_rmse = rmse(mean_mu_full)
    mlx_full_ll = float(np.mean(logsumexp(lps_full, axis=0) - np.log(S)))
    mlx_full_cov = coverage95(preds_full)
    mlx_full_crps = crps(preds_full)

print("\n=== Comparación en el conjunto de prueba ===")
hdr = "%-40s%9s%13s%10s%9s%10s%10s" % ("Modelo", "RMSE", "log-lik/pto", "cob.95%", "CRPS", "paso/seg", "grads/paso")
print(hdr)
print("%-40s%9.4f%13.4f%10s%9s%10s%10s" % ("MLP determinista (MAP)", det_rmse, det_ll, "-", "-", "-", "-"))
print("%-40s%9.4f%13.4f%10.3f%9.4f%10.1f%10.2f" % ("Bayes última capa (HMC-MLX)", mlx_rmse, mlx_ll, mlx_cov, mlx_crps, steps_per_sec_last, grads_per_step_last))
print("%-40s%9.4f%13.4f%10.3f%9.4f%10.1f%10.2f" % ("Bayes red completa (HMC-MLX)", mlx_full_rmse, mlx_full_ll, mlx_full_cov, mlx_full_crps, steps_per_sec_full, grads_per_step_full))
print("(log-lik mayor es mejor; CRPS menor es mejor; cobertura 95% ideal ≈ 0.95)")
print("Nota: la red completa tiene %d parámetros vs %d de la última capa." % (model_ndim_full, model_ndim_last))


# ---------------------------------------------------------------------------
# Gráfica: determinista vs bayesiano (última capa vs red completa) en la malla.
# ---------------------------------------------------------------------------
def grid_post_mean(samples):
    phi_g = np.asarray(phi_grid)
    mus = np.array([head_np(flat, phi_g)[0] for flat in samples]) * y_sd + y_mean
    return mus.mean(axis=0), np.percentile(mus, 2.5, axis=0), np.percentile(mus, 97.5, axis=0)


def grid_post_mean_full(samples, sizes):
    Xgrid_np = np.asarray(Xgrid)
    mus = np.array([predict_full_network_np(flat, Xgrid_np, sizes)[0] for flat in samples]) * y_sd + y_mean
    return mus.mean(axis=0), np.percentile(mus, 2.5, axis=0), np.percentile(mus, 97.5, axis=0)


det_grid = head_np(np.asarray(flat0), np.asarray(phi_grid))[0] * y_sd + y_mean
mlx_mean, mlx_lo, mlx_hi = grid_post_mean(samples_mlx)
mlx_full_mean, mlx_full_lo, mlx_full_hi = grid_post_mean_full(samples_mlx_full, sizes_full)

plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, s=8, alpha=0.2, color="gray", label="datos")
plt.plot(xx, f(xx), "k--", label="media verdadera")
plt.plot(xx, det_grid, "C1", label="MLP determinista (MAP)")
plt.plot(xx, mlx_mean, "C0", ls="--", lw=1.5, label="Bayes última capa (HMC-MLX)")
plt.plot(xx, mlx_full_mean, "C3", label="Bayes red completa (HMC-MLX)")
plt.fill_between(xx.ravel(), mlx_lo, mlx_hi, color="C0", alpha=0.15,
                 label="IC 95% (última capa)")
plt.fill_between(xx.ravel(), mlx_full_lo, mlx_full_hi, color="C3", alpha=0.15,
                 label="IC 95% (red completa)")
plt.legend()
plt.title("Determinista (MAP) vs Bayesiano: última capa vs red completa — HMC-MLX")
plt.tight_layout()
plt.savefig("bayesian_neural_net.png", dpi=120)
print("\nGráfica guardada en bayesian_neural_net.png")
