#  Copyright 2019-2020 George Ho
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Batched multi-chain Hamiltonian Monte Carlo using Apple's MLX framework.

This module provides :func:`sample_hmc_chains`, a self-contained multi-chain HMC
that advances *all* chains together as the leading axis of every array -- one
batched, on-device program per draw (a Python loop over the fixed-length
leapfrog path, MLX's lazy graph fused per step). It is the Apple-Silicon
(Metal) counterpart to a process-per-chain sampler: no OS processes, no
pickling, no per-chain Python loop.

The position ``q`` may be any MLX **pytree** (a single array, or an arbitrarily
nested dict/list/tuple of arrays). The leapfrog kernel is written entirely with
``mlx.utils.tree_map``, so the momentum, gradient and diagonal mass matrix all
share ``q``'s structure and a plain array is just the single-leaf case.

Two requirements follow from the design:

* The ``logp_dlogp_func`` is a **single-chain** ``q -> (logp, dlogp)`` built from
  ``mlx.core`` ops, where ``logp`` is a scalar MLX array and ``dlogp`` is an
  MLX-array pytree matching ``q``. It is batched over chains internally with
  ``mx.vmap``, so it must be MLX-traceable (no ``np.asarray`` / Python branching
  on values). Typically ``mx.value_and_grad(logp)``.
* Only **diagonal** mass matrices are supported (a per-leaf, per-dimension
  ``var`` pytree matching ``q``'s event shape, with the convention
  ``velocity = var * p`` and ``p ~ N(0, 1/var)``).

Divergence handling: a draw is flagged divergent when its final energy is
non-finite or the energy change exceeds ``Emax``.
"""

import numpy as np

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "blmx.hmc_mlx requires MLX. Install it with `pip install mlx` "
        "(Apple Silicon only)."
    ) from err


# Dual-averaging defaults (Hoffman & Gelman 2014, the usual Stan/PyMC values).
_DA_GAMMA = 0.05
_DA_T0 = 10.0
_DA_KAPPA = 0.75


def _leaves(tree):
    """The leaf arrays of a pytree (MLX ``tree_flatten`` yields ``(path, leaf)``)."""
    return [leaf for _, leaf in tree_flatten(tree)]


def _tree_n_chains(q):
    """Read the chain count from the leading axis of any leaf of ``q``."""
    return _leaves(q)[0].shape[0]


def _event_sum(leaf):
    """Sum a ``(chains, *event)`` leaf over its event axes -> ``(chains,)``."""
    if leaf.ndim <= 1:
        return leaf
    return mx.sum(leaf, axis=tuple(range(1, leaf.ndim)))


def _tree_dot(a, b):
    """Per-chain Euclidean inner product of two pytrees -> ``(chains,)``."""
    return sum(_leaves(tree_map(lambda x, y: _event_sum(x * y), a, b)))


def _kinetic_energy(p, var):
    """Diagonal kinetic energy ``0.5 * sum var * p**2`` per chain -> ``(chains,)``."""
    return 0.5 * sum(_leaves(tree_map(lambda pp, v: _event_sum(v * pp**2), p, var)))


def _tree_random_momentum(key, target, var):
    """Draw ``p ~ N(0, 1/var)`` as a pytree matching ``target``.

    One subkey per leaf gives every leaf independent noise. With the convention
    ``velocity = var * p``, the momentum has per-dimension variance ``1 / var``.
    """
    named = tree_flatten(target)
    keys = mx.random.split(key, len(named))
    var_leaves = _leaves(var)
    noise = [
        mx.random.normal(leaf.shape, key=k) / mx.sqrt(v)
        for k, (_, leaf), v in zip(keys, named, var_leaves)
    ]
    return tree_unflatten([(name, n) for (name, _), n in zip(named, noise)])


def _bcast(mask, leaf):
    """Reshape a ``(chains,)`` mask to broadcast against a ``(chains, *event)`` leaf."""
    return mask.reshape((mask.shape[0],) + (1,) * (leaf.ndim - 1))


def _where_tree(mask, a, b):
    """Per-chain ``where``: pick leaf-of-``a`` where ``mask`` else leaf-of-``b``."""
    return tree_map(lambda x, y: mx.where(_bcast(mask, x), x, y), a, b)


def _stack_traces(frames):
    """Stack a list of position pytrees into one pytree with a leading draw axis."""
    flat = [tree_flatten(f) for f in frames]
    names = [name for name, _ in flat[0]]
    stacked = [mx.stack([fr[i][1] for fr in flat], axis=0) for i in range(len(names))]
    return tree_unflatten(list(zip(names, stacked)))


def _first_leaf(tree):
    """The first leaf array of a pytree."""
    return _leaves(tree)[0]


def _make_move(batched_ldf, n_leapfrog, Emax):
    """Build a single fixed-path-length HMC transition over all chains.

    Returns ``move(q, key, step_size, var) -> (q_next, accept_prob, diverging)``
    where ``q`` leaves are ``(chains, *event)``; ``step_size`` and the diagonal
    mass ``var`` (event-shaped, shared across chains) broadcast over the batch.
    """

    def move(q, key, step_size, var):
        key_p, key_a = mx.random.split(key)
        logp0, grad0 = batched_ldf(q)
        p0 = _tree_random_momentum(key_p, q, var)
        energy0 = _kinetic_energy(p0, var) - logp0  # (chains,)

        dt = 0.5 * step_size
        q1, p, grad = q, p0, grad0
        for _ in range(n_leapfrog):
            p = tree_map(lambda pp, g: pp + dt * g, p, grad)
            q1 = tree_map(lambda qq, pp, v: qq + step_size * (v * pp), q1, p, var)
            _, grad = batched_ldf(q1)
            p = tree_map(lambda pp, g: pp + dt * g, p, grad)

        logp1, _ = batched_ldf(q1)
        energy1 = _kinetic_energy(p, var) - logp1

        energy_change = energy0 - energy1
        energy_change = mx.where(mx.isnan(energy_change), -mx.inf, energy_change)
        diverging = (~mx.isfinite(energy1)) | (mx.abs(energy_change) > Emax)

        accept_prob = mx.minimum(1.0, mx.exp(energy_change))  # (chains,)
        n_chains = energy0.shape[0]
        accepted = (~diverging) & (mx.random.uniform(shape=(n_chains,), key=key_a) < accept_prob)
        q_next = _where_tree(accepted, q1, q)
        return q_next, accept_prob, diverging

    return move


def _run_phase(move, target_accept, q0, key, var, n_steps, init_logstep, adapt):
    """Advance all chains ``n_steps`` draws, optionally dual-averaging the step size.

    Returns ``(q_final, positions, accept_prob, diverging, logstep_bar)`` where
    ``positions`` is a pytree matching ``q0`` with a leading ``(n_steps, chains, ...)``
    axis. A single step size shared across chains is tuned on the mean accept
    statistic; ``logstep_bar`` carries the averaged log step into the next phase.
    """
    mu_da = mx.log(10.0) + init_logstep
    logstep = init_logstep
    logstep_bar = init_logstep
    hbar = mx.array(0.0)
    t = 0.0

    q = q0
    frames, accs, divs = [], [], []
    for i in range(n_steps):
        key, sub = mx.random.split(key)
        step_size = mx.exp(logstep)
        q, accept_prob, diverging = move(q, sub, step_size, var)

        if adapt:
            t += 1.0
            w = 1.0 / (t + _DA_T0)
            hbar = (1.0 - w) * hbar + w * (target_accept - mx.mean(accept_prob))
            logstep = mu_da - (np.sqrt(t) / _DA_GAMMA) * hbar
            eta = t ** (-_DA_KAPPA)
            logstep_bar = eta * logstep + (1.0 - eta) * logstep_bar

        frames.append(q)
        accs.append(accept_prob)
        divs.append(diverging)
        # Materialize per draw so the lazy graph does not grow without bound.
        mx.eval(q, accept_prob, diverging, logstep, logstep_bar, hbar)

    positions = _stack_traces(frames)
    accept_prob = mx.stack(accs, axis=0)
    diverging = mx.stack(divs, axis=0)
    return q, positions, accept_prob, diverging, logstep_bar


def _init_start(start, model_ndim, chains, key):
    """Coerce ``start`` to a pytree whose leaves lead with a ``chains`` axis."""
    if start is None:
        return mx.random.normal((chains, model_ndim), key=key)

    def broadcast(leaf):
        leaf = mx.array(leaf)
        if leaf.shape[:1] == (chains,):
            return leaf
        return mx.broadcast_to(leaf, (chains,) + leaf.shape)

    return tree_map(broadcast, start)


def _finalize(positions, accept_prob, diverging, logstep_bar, var):
    """Pack device arrays into ArviZ-friendly NumPy ``(trace, stats)``."""
    # positions leaves: (draws, chains, *event) -> (chains, draws, *event).
    trace = tree_map(lambda pos: np.asarray(pos).swapaxes(0, 1), positions)
    stats = {
        "acceptance_rate": np.asarray(accept_prob).T,  # (chains, draws)
        "diverging": np.asarray(diverging).T.astype(bool),
        "step_size": float(mx.exp(logstep_bar).item()),
        "mass_matrix_inv": tree_map(np.asarray, var),
    }
    return trace, stats


def _adapt_and_sample(move, target_accept, q0, key, model_ndim, draws, tune,
                      init_step, adapt_mass):
    """Shared warmup-then-sample driver for the HMC and NUTS MLX backends.

    ``move`` is a single batched transition ``(q, key, step, var) -> (q', ap, div)``.
    Warmup tunes one shared step size (dual averaging) and, if ``adapt_mass``,
    a diagonal mass matrix from the first window's variance; both freeze for the
    sampling phase.
    """
    dtype = _first_leaf(q0).dtype
    init_logstep = mx.log(mx.array(init_step, dtype=dtype))
    # Diagonal mass matrix as event-shaped ones (shared across chains).
    var = tree_map(lambda leaf: mx.ones(leaf.shape[1:], dtype=leaf.dtype), q0)
    logstep_bar = init_logstep

    run = lambda *a: _run_phase(move, target_accept, *a)

    if tune > 0:
        n_w1 = tune // 2 if adapt_mass else tune
        key, sub = mx.random.split(key)
        q0, w1_pos, _, _, logstep_bar = run(q0, sub, var, n_w1, init_logstep, True)
        if adapt_mass and tune - n_w1 > 0:
            # Diagonal mass matrix from the second half of window 1.
            var = tree_map(
                lambda pos: mx.clip(mx.var(pos[n_w1 // 2:], axis=(0, 1)), 1e-8, None),
                w1_pos,
            )
            key, sub = mx.random.split(key)
            q0, _, _, _, logstep_bar = run(q0, sub, var, tune - n_w1, logstep_bar, True)

    key, sub = mx.random.split(key)
    _, positions, accept_prob, diverging, _ = run(q0, sub, var, draws, logstep_bar, False)
    return _finalize(positions, accept_prob, diverging, logstep_bar, var)


def sample_hmc_chains(
    logp_dlogp_func,
    model_ndim,
    draws=1000,
    tune=1000,
    chains=4,
    n_leapfrog=16,
    target_accept=0.8,
    init_step=0.1,
    Emax=1000.0,
    adapt_mass=True,
    random_seed=0,
    start=None,
):
    """Sample multiple chains with a batched, on-device fixed-path-length HMC (MLX).

    All ``chains`` advance together as the leading array axis. Warmup tunes a
    single (shared) step size by dual averaging and, if ``adapt_mass`` is True, a
    diagonal mass matrix estimated from the warmup draws; both are then frozen for
    the sampling phase.

    Parameters
    ----------
    logp_dlogp_func : callable
        Single-chain, MLX-traceable ``q -> (logp, dlogp)`` (scalar ``logp``,
        ``dlogp`` a pytree matching ``q``); batched over chains internally with
        ``mx.vmap``. Typically ``mx.value_and_grad(logp)``.
    model_ndim : int
        Length of ``q`` in the flat-array default case. Used only when ``start``
        is None; ignored for pytree inputs, which must be supplied via ``start``.
    draws, tune : int
        Post-warmup draws and warmup steps. Warmup is split in half when
        ``adapt_mass`` is True (estimate mass, then re-tune the step).
    chains : int
        Number of chains (the batch size).
    n_leapfrog : int
        Fixed number of leapfrog steps per proposal.
    target_accept : float
        Target mean acceptance for dual-averaging step-size adaptation.
    init_step : float
        Initial leapfrog step size before adaptation.
    Emax : float
        Energy-change threshold above which a draw is flagged divergent.
    adapt_mass : bool
        If True, estimate a diagonal mass matrix from the first warmup window.
    random_seed : int
        Seed for ``mx.random.key``.
    start : pytree, optional
        Initial positions; either ``(chains, model_ndim)`` / ``(model_ndim,)``
        (broadcast across chains), or a pytree whose leaves lead with a ``chains``
        axis (or broadcast to one). Defaults to standard-normal draws.

    Returns
    -------
    trace : pytree of np.ndarray
        Posterior draws, a pytree matching the position with each leaf of shape
        ``[chains, draws, *event]``.
    stats : dict
        ``acceptance_rate`` and ``diverging`` (each ``[chains, draws]``), plus the
        frozen ``step_size`` (float) and ``mass_matrix_inv`` (a pytree).
    """
    key = mx.random.key(random_seed)
    key, key_init = mx.random.split(key)
    q0 = _init_start(start, model_ndim, chains, key_init)
    batched_ldf = mx.vmap(logp_dlogp_func)
    move = _make_move(batched_ldf, int(n_leapfrog), Emax)
    return _adapt_and_sample(
        move, target_accept, q0, key, model_ndim, draws, tune, init_step, adapt_mass
    )
