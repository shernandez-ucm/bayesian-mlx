"""bayesian-mlx: batched multi-chain HMC and NUTS samplers on Apple's MLX.

Both samplers advance all chains together as the leading axis of every array,
running one batched, on-device (Metal) program per draw -- no OS processes, no
pickling, no per-chain Python loop. Positions may be any MLX pytree, and the
``logp_dlogp_func`` is a single-chain, MLX-traceable ``q -> (logp, dlogp)``
(typically ``mx.value_and_grad(logp)``), batched over chains with ``mx.vmap``.
"""

from .hmc_mlx import sample_hmc_chains
from .nuts_mlx import sample_nuts_chains

__all__ = ["sample_hmc_chains", "sample_nuts_chains"]
