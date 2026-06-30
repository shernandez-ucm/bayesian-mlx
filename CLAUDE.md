# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two multi-chain MCMC samplers — HMC and NUTS — implemented on **Apple's MLX framework** (Metal / Apple Silicon), plus example scripts that exercise them and compare results against **PyMC's NUTS**.

- `blmx/hmc_mlx.py` → `sample_hmc_chains` (fixed-path-length HMC)
- `blmx/nuts_mlx.py` → `sample_nuts_chains` (adaptive No-U-Turn)
- `blmx/__init__.py` re-exports both: `from blmx import sample_hmc_chains, sample_nuts_chains`

The import package is `blmx`. There is no longer any dependency on (or reference to) `littlemcmc` or JAX — this was ported from a JAX implementation, but all of that is gone.

## Setup & running

MLX requires Apple Silicon. The examples additionally need PyMC/ArviZ (the reference) and matplotlib (the BNN plot).

```
python3 -m venv .venv
.venv/bin/pip install -e ".[examples]"     # blmx + mlx + numpy, plus pymc/arviz/matplotlib
```

There is no test runner; verification is by running the examples (all compare MLX output to PyMC NUTS):

```
.venv/bin/python examples/gaussian_mixture_samplers.py   # HMC/NUTS/PyMC on a 2D bimodal target
.venv/bin/python examples/benchmark_8_schools.py         # pytree params; ArviZ summaries vs PyMC
.venv/bin/python examples/benchmark_mlx_vs_pymc.py --dim 30 --chains 2,4,8   # throughput + recovery
.venv/bin/python examples/bayesian_neural_net.py         # last-layer Bayes: NUTS-MLX vs NUTS-PyMC
```

Each example inserts the repo root on `sys.path`, so they also run without an editable install. PyMC prints progress to stderr; it is noisy but harmless.

## Architecture

**Chains are the leading axis of every array**, not OS processes. Every chain advances together inside one batched, on-device (Metal) program; MLX's lazy graph fuses each step. This is the Apple-Silicon analogue of the GPU-vmap design the JAX version had, but **MLX has no `jax.jit`/`lax.scan`/`lax.while_loop`/`lax.cond`** — so the structure is fundamentally different from the old code:

- **The draws loop is a plain Python `for` loop.** `_run_phase` (in `hmc_mlx.py`) iterates draws, calls the per-draw `move`, runs dual-averaging in Python, and calls `mx.eval(...)` once per draw so the lazy graph doesn't grow without bound. This is the single most important MLX idiom here — without the per-draw `eval`, memory blows up.
- **HMC's fixed leapfrog path is an unrolled Python loop** inside `move`.
- **NUTS builds each subtree iteratively** (`_build_subtree`), via NumPyro's "Iterative NUTS" scheme (https://github.com/pyro-ppl/numpyro/wiki/Iterative-NUTS): a flat Python loop takes leaves one leapfrog step at a time and folds them into a running subtree, while an explicit checkpoint stack (popcount/trailing-ones bit tricks on the leaf index) reproduces every U-turn check bottom-up recursion would perform. Because MLX is eager, the leaf index is a concrete Python int, so the checkpoint stack is a plain Python list — no on-device dynamic-index updates needed (simpler than the JAX original, which needs `lax.while_loop` + traced array indices precisely because it lacks real recursion).

### Manual chain-batching (not `mx.vmap` over the trajectory)
The *user's* `logp_dlogp_func` is batched over chains with `mx.vmap` (`batched_ldf = mx.vmap(logp_dlogp_func)` in both samplers). But the **sampler kernels themselves do not vmap** — they operate on arrays whose leading axis is the chain, and reduce only over event axes:
- `_event_sum` / `_tree_dot` / `_kinetic_energy` sum over event axes (`axis=1..`) and keep the `(chains,)` axis. Per-chain scalars (energy, accept prob, U-turn angles, turning/diverging flags) are all `(chains,)` arrays.
- `_bcast(mask, leaf)` reshapes a `(chains,)` mask to broadcast against a `(chains, *event)` leaf; `_where_tree` uses it for per-chain selection.

This manual batching (rather than `mx.vmap` of the whole transition) is **required for NUTS**: the trajectory has data-dependent control flow (Python loops/recursion that branch on `.item()`), which `mx.vmap` cannot trace. Different chains U-turn at different depths, so NUTS carries a per-chain `active` mask and `_freeze`s chains that have terminated while the rest keep doubling; the outer loop stops once `mx.any(active).item()` is false.

### The shared `move` / `_adapt_and_sample` contract (the key seam)
Both samplers build a single batched transition `move(q, key, step_size, var) -> (q_next, accept_prob, diverging)` and hand it to the **shared** `_adapt_and_sample` driver in `hmc_mlx.py` (which NUTS imports). That driver owns warmup, dual-averaging, the two-window mass-matrix estimate, and finalization — so HMC and NUTS share one adaptation/scan/output path. **If you add a sampler, match this exact `move` signature and return tuple** and you get warmup + output for free. `nuts_mlx.py` imports `_adapt_and_sample`, `_kinetic_energy`, `_tree_random_momentum`, `_tree_dot`, `_where_tree`, `_bcast`, `_init_start` from `hmc_mlx.py`; keep shared conventions in `hmc_mlx.py`.

### Conventions inherited from the original design
- **`logp_dlogp_func` is single-chain and MLX-traceable**: `q -> (logp, dlogp)`, scalar `logp`, `dlogp` a pytree matching `q`, built from `mlx.core` ops (no `np.asarray`, no Python branching on values). Typically `mx.value_and_grad(logp)`.
- **Diagonal mass matrix only** — a per-leaf, event-shaped `var` pytree (shared across chains), convention `velocity = var * p`, `p ~ N(0, 1/var)`.
- **Pytree positions**: `q` may be any MLX pytree; a flat array is the single-leaf case. NB: MLX's `mlx.utils.tree_flatten` returns a flat `[(path, leaf), ...]` list (**not** JAX's `(leaves, treedef)` tuple) — the `_leaves` helper exists because of this.
- **Output contract**: `(trace, stats)`. `trace` is a pytree matching the position, each leaf `(chains, draws, *event)` (ArviZ-friendly). `stats` has `acceptance_rate`, `diverging` (each `(chains, draws)`), plus scalar `step_size` and `mass_matrix_inv` pytree.
- **Warmup**: dual-average a single shared step size; if `adapt_mass`, split `tune` in half — estimate a diagonal mass matrix from window-1 variance, then re-tune. Both freeze for sampling.

### NUTS trajectory specifics (`nuts_mlx.py`)
Multinomial / biased-progressive trajectory sampling with the generalized U-turn criterion (Betancourt 2017; Hoffman & Gelman 2014 Alg. 6). `_Tree` is a small `__slots__` class (not a pytree — handled field-by-field so `tree_map` never traverses it). `_combine` merges two subtrees, taking outer boundaries from the right side per chain, and resampling the proposal with a biased kernel (between main trees, prob 0 if the new half turns/diverges) or a uniform/multinomial kernel (within a subtree); it also computes the cross-subtree U-turn (`check_turning=True`, the default) — but only between **equal-size, aligned** halves, which holds for the outer doubling combine. `_build_subtree`'s leaf-by-leaf fold combines a running (generally unaligned) prefix with one new leaf, so it passes `check_turning=False` and checks the generalized U-turn explicitly via the checkpoint stack instead (`_leaf_idx_to_ckpt_idxs`). A divergent or turning subtree is never selected as the next proposal.

### Performance note
NUTS runs in eager Python (no compile across the dynamic trajectory), so it is **noticeably slower than the fixed-path HMC** (which is a static unrolled loop and could be `mx.compile`d). This is an inherent MLX tradeoff for data-dependent control flow, not a bug — see `benchmark_mlx_vs_pymc.py`.

## Conventions

- `examples/bayesian_neural_net.py` comments/prints are in **Spanish**; match that language when editing it. The library modules and other examples are in English.
- Examples run in **float32** (MLX default); input data is cast to `float32`.
- MLX has no `lgamma`; `bayesian_neural_net.py` implements a differentiable Lanczos `lgamma` for the Student-t normalizing constant (verified against `scipy.special.gammaln`). If you need other special functions, expect to hand-roll them.
- The examples compare against modern **PyMC** (v6, the successor to PyMC3). The reference model is always built to match the MLX log-density exactly (same prior, same likelihood) so the two posteriors should agree.
