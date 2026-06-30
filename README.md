# Bayesian MCMC on MLX

High-performance MCMC samplers — **HMC** and **NUTS** — implemented on **Apple's MLX framework** (Metal / Apple Silicon).

This project provides efficient Bayesian inference tools optimized for Apple's neural processing capabilities, with multi-chain sampling that executes as a single batched, on-device program.

## Features

- **HMC Sampler** (`blmx.sample_hmc_chains`) — Fixed-path-length Hamiltonian Monte Carlo
- **NUTS Sampler** (`blmx.sample_nuts_chains`) — Adaptive No-U-Turn sampling with trajectory building via recursion
- **Multi-chain batching** — All chains advance together in a single on-device computation graph
- **Automatic adaptation** — Dual-averaging step-size tuning and optional diagonal mass-matrix estimation
- **PyMC compatibility** — Reference implementations and comparison tools included

## Requirements

- **Apple Silicon** (M1, M2, M3, etc.) — MLX requires Metal
- **Python 3.9+**
- **MLX**, **NumPy**, and core dependencies
- **PyMC, ArviZ, Matplotlib** — for examples and comparisons (optional)

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[examples]"
```

This installs:
- Core library (`blmx`) with MLX and NumPy
- Example dependencies (PyMC, ArviZ, Matplotlib)

## Quick Start

```python
import mlx.core as mx
from blmx import sample_hmc_chains, sample_nuts_chains

# Define your target density
def logp(q):
    return -0.5 * mx.sum(q**2)

# HMC sampling
trace, stats = sample_hmc_chains(
    logp_dlogp_func=mx.value_and_grad(logp),
    num_chains=4,
    num_draws=2000,
    num_warmup=1000,
    key=mx.random.key(0)
)

# NUTS sampling
trace, stats = sample_nuts_chains(
    logp_dlogp_func=mx.value_and_grad(logp),
    num_chains=4,
    num_draws=2000,
    num_warmup=1000,
    key=mx.random.key(0)
)
```

## Examples

All examples compare MLX output against PyMC's NUTS reference implementation:

```bash
# 2D bimodal target — visualize HMC, NUTS, and PyMC samples
.venv/bin/python examples/gaussian_mixture_samplers.py

# Pytree parameters; ArviZ posterior summaries vs PyMC
.venv/bin/python examples/benchmark_8_schools.py

# Throughput + posterior recovery at multiple chain/dim scales
.venv/bin/python examples/benchmark_mlx_vs_pymc.py --dim 30 --chains 2,4,8

# Bayesian neural network — last-layer inference (NUTS-MLX vs NUTS-PyMC)
.venv/bin/python examples/bayesian_neural_net.py
```

## Architecture

### Chain Batching Strategy
- **Not `mx.vmap` of the trajectory** — chains run together as the leading axis of every array
- **Python `for` loops drive the sampler** — one call to `mx.eval()` per draw keeps the lazy graph bounded
- **Per-chain scalars** — energy, acceptance, divergence, and U-turn indicators are `(chains,)` arrays
- **HMC:** Fixed leapfrog path is an unrolled Python loop
- **NUTS:** Trajectory built by recursion (`_build_subtree`), with per-chain active mask for dynamic termination

### Shared Adaptation Pipeline
Both samplers use the same `_adapt_and_sample` driver (in `hmc_mlx.py`), providing:
- Dual-averaging step-size tuning
- Optional diagonal mass-matrix learning (two-window approach)
- Frozen adaptation for sampling phase
- Unified output format

### API Contract for New Samplers
To add a sampler, implement:
```python
def move(q, key, step_size, var) -> (q_next, accept_prob, diverging)
```
Pass it to `_adapt_and_sample` and get warmup + output handling for free.

## Output Format

Both samplers return `(trace, stats)`:

- **`trace`** — PyTree matching position shape; each leaf is `(chains, draws, *event_dims)`
  - ArviZ-compatible format
- **`stats`** — Diagnostics dictionary:
  - `acceptance_rate` — `(chains, draws)`
  - `diverging` — `(chains, draws)` boolean divergence flags
  - `step_size` — scalar (shared across chains)
  - `mass_matrix_inv` — PyTree of per-leaf inverse mass matrices

## Performance Notes

- **NUTS** runs eager Python recursion (no compile across dynamic trajectory) → slower than fixed-path HMC
- **HMC** is a static unrolled loop → could be wrapped with `mx.compile` for additional speedup
- This is an inherent MLX tradeoff for data-dependent control flow, not a deficiency

See `examples/benchmark_mlx_vs_pymc.py` for throughput comparisons.

## Conventions

- **Diagonal mass matrices only** — per-leaf, event-shaped `var` PyTree
- **Velocity parameterization:** `velocity = var * p`, with `p ~ N(0, I / var)`
- **PyTree positions:** `q` can be any MLX PyTree; scalar/vector are single-leaf cases
- **No special functions:** MLX lacks `lgamma`; `bayesian_neural_net.py` includes a differentiable Lanczos implementation
- **Language:** Examples are in English; `bayesian_neural_net.py` contains Spanish comments
- **Precision:** Examples run in float32 (MLX default); input data is cast accordingly

## Project Structure

```
blmx/
├── __init__.py           # Public API exports
├── hmc_mlx.py            # HMC sampler + shared adaptation driver
└── nuts_mlx.py           # NUTS sampler (uses shared driver from HMC)

examples/
├── gaussian_mixture_samplers.py   # 2D bimodal visualization
├── benchmark_8_schools.py          # Pytree parameters demo
├── benchmark_mlx_vs_pymc.py        # Throughput & recovery
└── bayesian_neural_net.py          # Last-layer Bayesian inference

bayesian_neural_net.png             # Example output visualization
```

## Design Decisions

1. **Manual chain-batching** instead of `mx.vmap` over the trajectory
   - NUTS has data-dependent control flow; `mx.vmap` cannot trace it
   - Each chain can U-turn at a different depth; we use per-chain `active` masks

2. **Recursion for NUTS trajectory**
   - No JIT tracing constraint means recursion is natural and checkpoint-free
   - Eliminates tree-doubling machinery, bit tricks, and saved checkpoints from the old JAX version

3. **Per-draw `mx.eval()`**
   - Prevents unbounded growth of the lazy computation graph
   - Most important MLX idiom in this codebase

4. **Shared `_adapt_and_sample` pipeline**
   - DRY principle: HMC and NUTS share warmup, mass-matrix estimation, and output formatting
   - New samplers inherit these for free

## References

- Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." *arXiv:1701.02434*
- Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." JMLR, 15.
- [MLX Documentation](https://ml-explore.github.io/mlx)
- [PyMC Documentation](https://docs.pymc.io)

## License

See LICENSE file (if present) or contact the author.

## Author

Sergio Hernandez (<shernandez.ucm@gmail.com>)
