"""Benchmark: MLX batched-chain samplers vs PyMC's process-parallel NUTS.

The MLX samplers advance all chains together as the leading axis of every array,
so the whole multi-chain update runs as one batched, on-device (Metal) program --
no OS processes, no pickling, no Python loop per draw. PyMC instead runs one
chain per worker process. This script measures wall-clock throughput of both as
the number of chains grows, on a shared target: a ``D``-dimensional standard
normal (identity mass matrix is exact, so every sampler is correct and we can
check recovery of ``mean = 0`` / ``std = 1``).

Caveats (an honest throughput comparison, not an identical-FLOPs one):

* PyMC's row is full NUTS with step-size and diagonal-mass adaptation and an
  adaptive (per-draw) path length, across worker processes.
* The MLX HMC row (``sample_hmc_chains``) uses a *fixed* number of leapfrog steps
  (the batch needs a uniform trip count), with dual-averaging step size and
  diagonal mass adaptation. Its reported time includes one-off Metal compilation.
* The MLX NUTS row (``sample_nuts_chains``) uses the adaptive No-U-Turn path
  length (recursive tree doubling, capped at ``max_treedepth``), so its per-draw
  work varies with the trajectory. Its time also includes one-off compilation.

So the samplers do slightly different per-draw work; the comparison is "valid
draws per second of each strategy", plus a correctness check.

Usage::

    python examples/benchmark_mlx_vs_pymc.py
    python examples/benchmark_mlx_vs_pymc.py --dim 50 --chains 2,4,8,16
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_pymc(dim, chains, draws, tune, seed):
    """Time PyMC's process-parallel NUTS over ``chains`` chains."""
    import pymc as pm

    with pm.Model():
        pm.Normal("x", 0.0, 1.0, shape=dim)
        t0 = time.perf_counter()
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=chains,
            progressbar=False, random_seed=seed,
        )
        elapsed = time.perf_counter() - t0

    samples = np.asarray(idata.posterior["x"]).reshape(-1, dim)
    return {
        "elapsed": elapsed,
        "draws_per_s": chains * draws / elapsed,
        "std": float(samples.std(axis=0).mean()),
    }


def benchmark_mlx(sampler, dim, chains, draws, tune, seed, **kw):
    """Time an MLX batched-chain sampler over ``chains`` chains (incl. compile)."""
    import mlx.core as mx

    def logp(x):
        return -0.5 * mx.sum(x * x)

    ldf = mx.value_and_grad(logp)
    t0 = time.perf_counter()
    trace, stats = sampler(
        ldf, dim, draws=draws, tune=tune, chains=chains, random_seed=seed, **kw
    )
    elapsed = time.perf_counter() - t0  # includes one-off Metal compilation

    samples = trace.reshape(-1, dim)
    return {
        "elapsed": elapsed,
        "draws_per_s": chains * draws / elapsed,
        "std": float(samples.std(axis=0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=30, help="target dimensionality")
    parser.add_argument("--chains", type=str, default="2,4,8,16",
                        help="comma-separated chain counts")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--leapfrog", type=int, default=10, help="MLX HMC leapfrog steps")
    parser.add_argument("--max-treedepth", type=int, default=10, help="MLX NUTS max tree depth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from blmx import sample_hmc_chains, sample_nuts_chains

    chain_counts = [int(c) for c in args.chains.split(",")]

    print("Target: %d-D standard normal | draws=%d tune=%d | MLX L=%d max_treedepth=%d"
          % (args.dim, args.draws, args.tune, args.leapfrog, args.max_treedepth))
    print("%-7s | %-26s | %-26s | %-26s"
          % ("chains", "pymc (process-parallel)", "mlx-hmc (batched)", "mlx-nuts (batched)"))
    print("%-7s | %-9s %-7s %-7s | %-9s %-7s %-7s | %-9s %-7s %-7s"
          % ("", "time(s)", "draw/s", "std", "time(s)", "draw/s", "std",
             "time(s)", "draw/s", "std"))
    print("-" * 96)

    for chains in chain_counts:
        pmr = benchmark_pymc(args.dim, chains, args.draws, args.tune, args.seed)
        hmr = benchmark_mlx(sample_hmc_chains, args.dim, chains, args.draws, args.tune,
                            args.seed, n_leapfrog=args.leapfrog)
        nur = benchmark_mlx(sample_nuts_chains, args.dim, chains, args.draws, args.tune,
                            args.seed, max_treedepth=args.max_treedepth)
        print("%-7d | %-9.3f %-7.0f %-7.3f | %-9.3f %-7.0f %-7.3f | %-9.3f %-7.0f %-7.3f"
              % (chains,
                 pmr["elapsed"], pmr["draws_per_s"], pmr["std"],
                 hmr["elapsed"], hmr["draws_per_s"], hmr["std"],
                 nur["elapsed"], nur["draws_per_s"], nur["std"]))

    print("-" * 96)
    print("std should be ~1.0 for all (recovering the standard normal).")
    print("MLX 'time(s)' includes one-off Metal compilation and the warmup/tuning phase.")


if __name__ == "__main__":
    main()
