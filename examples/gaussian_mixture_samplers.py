"""Sample a 2D Gaussian mixture with the MLX samplers, checked against PyMC NUTS.

This exercises, on a single shared target:

* ``blmx.sample_hmc_chains``   -- the batched multi-chain MLX HMC
* ``blmx.sample_nuts_chains``  -- the batched multi-chain MLX NUTS
* PyMC's NUTS                  -- the reference, on the *same* log-density

The target is a balanced mixture of two isotropic 2D Gaussians with means at
``(-2, 0)`` and ``(2, 0)``. The barrier between the modes is shallow enough that
the chains mix across both, so each sampler should recover the full bimodal
marginal. For every sampler we pool all chains and report:

* the marginal mean        -- target ``[0, 0]`` (the mixture is symmetric),
* the marginal std         -- target ``[sqrt(5), 1] ~= [2.236, 1.0]``,
* the mode coverage        -- fraction of samples on each side of ``x0 = 0``
  (target ``0.5`` / ``0.5``) and the conditional mean of each side (target
  ``-2`` and ``+2``), i.e. both modes are actually populated,
* the mean acceptance statistic and the number of post-tuning divergences,
* the wall-clock sampling time (incl. warmup; for MLX, the first call also pays
  one-off Metal kernel compilation).

Run with ``python examples/gaussian_mixture_samplers.py``. Requires ``mlx`` (the
samplers) and ``pymc``/``arviz`` (the reference).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from blmx import sample_hmc_chains, sample_nuts_chains

# --- Target: balanced mixture of two isotropic 2D Gaussians (identity cov) -----
WEIGHTS = np.array([0.5, 0.5])
MEANS = np.array([[-2.0, 0.0], [2.0, 0.0]])
MODEL_NDIM = 2

# Sampling settings shared across samplers.
DRAWS = 1000
TUNE = 1000
CHAINS = 4
SEED = 42


def mlx_logp(x):
    """MLX-traceable mixture log-density (scalar) for one chain's position ``x``."""
    means = mx.array(MEANS.astype(np.float32))
    weights = mx.array(WEIGHTS.astype(np.float32))
    diff = x[None, :] - means              # (n_components, ndim)
    sq = mx.sum(diff**2, axis=1)           # (n_components,)
    log_comp = mx.log(weights) - 0.5 * sq - mx.log(2 * np.pi)
    return mx.logsumexp(log_comp)


# Single-chain (logp, grad); the samplers batch this over chains with mx.vmap.
mlx_logp_dlogp = mx.value_and_grad(mlx_logp)


def summarize(name, trace, accept, n_div, elapsed):
    """Print pooled-chain diagnostics for one sampler run."""
    samples = trace.reshape(-1, trace.shape[-1])  # (chains * draws, ndim)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)

    x0 = samples[:, 0]
    frac_left = float(np.mean(x0 < 0))
    left_mean = float(x0[x0 < 0].mean()) if np.any(x0 < 0) else float("nan")
    right_mean = float(x0[x0 >= 0].mean()) if np.any(x0 >= 0) else float("nan")

    print("=" * 70)
    print("%s" % name)
    print("-" * 70)
    print("trace shape          : %s" % (trace.shape,))
    print("marginal mean        : [% .3f, % .3f]   target [ 0, 0]" % (mean[0], mean[1]))
    print("marginal std         : [% .3f, % .3f]   target [ 2.236, 1.0]" % (std[0], std[1]))
    print(
        "mode coverage (L/R)  : %.2f / %.2f          target 0.50 / 0.50"
        % (frac_left, 1 - frac_left)
    )
    print("mode means (L/R)     : [% .3f, % .3f]   target [-2, +2]" % (left_mean, right_mean))
    print("mean accept stat     : %.3f" % accept)
    print("divergences          : %d" % n_div)
    print("wall time            : %.3f s  (incl. warmup & any one-off compile)" % elapsed)


def run_mlx(name, sampler):
    start = time.perf_counter()
    trace, stats = sampler(
        mlx_logp_dlogp,
        MODEL_NDIM,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=SEED,
    )
    elapsed = time.perf_counter() - start
    summarize(
        name, trace, float(np.mean(stats["acceptance_rate"])),
        int(np.sum(stats["diverging"])), elapsed,
    )


def run_pymc():
    import pymc as pm
    import pytensor.tensor as pt

    means = pt.constant(MEANS)
    log_weights = pt.constant(np.log(WEIGHTS))

    with pm.Model() as model:
        x = pm.Flat("x", shape=MODEL_NDIM)
        diff = x[None, :] - means
        sq = pt.sum(diff**2, axis=1)
        log_comp = log_weights - 0.5 * sq - np.log(2 * np.pi)
        pm.Potential("like", pt.logsumexp(log_comp))

        start = time.perf_counter()
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=1,
            target_accept=0.8,
            initvals={"x": np.array([2.0, 0.0])},
            random_seed=SEED,
            progressbar=False,
        )
        elapsed = time.perf_counter() - start

    trace = np.asarray(idata.posterior["x"])  # (chains, draws, ndim)
    accept = float(np.mean(idata.sample_stats["acceptance_rate"].values))
    n_div = int(idata.sample_stats["diverging"].values.sum())
    summarize("PyMC NUTS (reference)", trace, accept, n_div, elapsed)


def main():
    run_mlx("sample_hmc_chains (MLX)", sample_hmc_chains)
    run_mlx("sample_nuts_chains (MLX)", sample_nuts_chains)
    run_pymc()


if __name__ == "__main__":
    main()
