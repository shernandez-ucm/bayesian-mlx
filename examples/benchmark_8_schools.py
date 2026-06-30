"""Eight schools (non-centered), sampled with the MLX samplers and checked against PyMC.

The parameters are kept as a **pytree** ``{"eta": (8,), "mu": (), "log_tau": ()}``
and sampled directly -- ``sample_hmc_chains`` / ``sample_nuts_chains`` run their
kernels leafwise via ``tree_map``, so there is no flat-vector packing. ``tau`` must
be positive, so we sample ``log_tau`` (unconstrained, which is what HMC/NUTS need);
with ``tau ~ LogNormal(0, 1)`` the implied density on ``log_tau`` is exactly
``Normal(0, 1)`` (the change-of-variables Jacobian is already folded in), which is
the prior we put on ``log_tau`` below.

All chains are the leading axis of every MLX array: every chain advances in
lockstep inside one batched, on-device program. We compare the MLX HMC and NUTS
posteriors against PyMC's NUTS on the equivalent model via ArviZ summaries.
"""

import os
import sys

import arviz as az
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from blmx import sample_hmc_chains, sample_nuts_chains

y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)
J = len(y)

# --- Sampler settings ---------------------------------------------------------
N_CHAINS = 4
N_WARMUP = 1000
N_DRAWS = 1000
N_LEAPFROG = 16
MAX_TREEDEPTH = 10
TARGET_ACCEPT = 0.9
SEED = 42
INIT_STEP = 0.1

_Y = mx.array(y)
_SIGMA = mx.array(sigma)
_LOG2PI = float(np.log(2 * np.pi))


def _normal_logpdf(x, loc, scale):
    z = (x - loc) / scale
    return -0.5 * (z * z + _LOG2PI) - mx.log(scale)


def logp(q):
    """MLX-traceable joint log-density of the non-centered eight-schools model.

    ``q`` is the sampled pytree in unconstrained space: {"eta", "mu", "log_tau"}.
    """
    eta, mu, log_tau = q["eta"], q["mu"], q["log_tau"]
    tau = mx.exp(log_tau)
    lp_eta = mx.sum(_normal_logpdf(eta, 0.0, 1.0))
    lp_mu = _normal_logpdf(mu, 0.0, 10.0)
    lp_log_tau = _normal_logpdf(log_tau, 0.0, 1.0)  # tau ~ LogNormal(0, 1)
    theta = mu + tau * eta
    lp_like = mx.sum(_normal_logpdf(_Y, theta, _SIGMA))
    return lp_eta + lp_mu + lp_log_tau + lp_like


# Single-chain (logp, grad); the samplers batch this over chains with mx.vmap.
logp_dlogp_func = mx.value_and_grad(logp)


def _start(seed):
    """Per-chain standard-normal starting points, one leaf per parameter."""
    k = mx.random.key(seed)
    k_eta, k_mu, k_tau = mx.random.split(k, 3)
    return {
        "eta": mx.random.normal((N_CHAINS, J), key=k_eta),
        "mu": mx.random.normal((N_CHAINS,), key=k_mu),
        "log_tau": mx.random.normal((N_CHAINS,), key=k_tau),
    }


def _to_idata(trace, stats):
    """Build an ArviZ InferenceData from a sampled pytree trace + stats."""
    log_tau = trace["log_tau"]
    posterior = {
        "eta": trace["eta"],
        "mu": trace["mu"],
        "log_tau": log_tau,
        "tau": np.exp(log_tau),
    }
    sample_stats = {
        "acceptance_rate": stats["acceptance_rate"],
        "diverging": stats["diverging"],
    }
    return az.from_dict({"posterior": posterior, "sample_stats": sample_stats})


def run_mlx_hmc():
    trace, stats = sample_hmc_chains(
        logp_dlogp_func, None,
        draws=N_DRAWS, tune=N_WARMUP, chains=N_CHAINS, n_leapfrog=N_LEAPFROG,
        target_accept=TARGET_ACCEPT, init_step=INIT_STEP, random_seed=SEED,
        start=_start(SEED),
    )
    return _to_idata(trace, stats), stats


def run_mlx_nuts():
    trace, stats = sample_nuts_chains(
        logp_dlogp_func, None,
        draws=N_DRAWS, tune=N_WARMUP, chains=N_CHAINS, max_treedepth=MAX_TREEDEPTH,
        target_accept=TARGET_ACCEPT, init_step=INIT_STEP, random_seed=SEED,
        start=_start(SEED),
    )
    return _to_idata(trace, stats), stats


def run_pymc():
    import pymc as pm

    with pm.Model():
        mu = pm.Normal("mu", 0.0, 10.0)
        tau = pm.LogNormal("tau", 0.0, 1.0)
        eta = pm.Normal("eta", shape=J)
        pm.Normal("obs", mu + tau * eta, sigma, observed=y)
        idata = pm.sample(
            draws=N_DRAWS, tune=N_WARMUP, chains=N_CHAINS, cores=1,
            target_accept=TARGET_ACCEPT, random_seed=SEED, progressbar=False,
        )
    return idata


def _report(name, idata, stats=None):
    print("\n%s" % name)
    print(az.summary(idata, var_names=["mu", "tau"], round_to=3).to_string())
    if stats is not None:
        print(
            "  mean accept: %.3f | divergences: %d | step size: %.4f"
            % (
                float(np.mean(stats["acceptance_rate"])),
                int(np.sum(stats["diverging"])),
                stats["step_size"],
            )
        )


def main():
    idata_hmc, stats_hmc = run_mlx_hmc()
    idata_nuts, stats_nuts = run_mlx_nuts()
    idata_pm = run_pymc()

    print("=" * 72)
    print("Eight schools -- posterior of mu and tau")
    print("(LogNormal(0,1) prior on tau; the three samplers should agree)")
    print("=" * 72)
    _report("MLX HMC", idata_hmc, stats_hmc)
    _report("MLX NUTS", idata_nuts, stats_nuts)
    _report("PyMC NUTS (reference)", idata_pm)


if __name__ == "__main__":
    main()
