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

"""Batched multi-chain No-U-Turn Sampler using Apple's MLX framework.

This module is the NUTS counterpart of :mod:`blmx.hmc_mlx`. It provides
:func:`sample_nuts_chains`, a self-contained multi-chain NUTS that advances *all*
chains together as the leading array axis, replacing ``hmc_mlx``'s fixed leapfrog
path length with the adaptive No-U-Turn termination of "Algorithm 6" of Hoffman &
Gelman (2014), using the multinomial/biased-progressive trajectory sampling and
generalized U-turn criterion of Betancourt (2017).

Because MLX uses ordinary Python control flow (no graph-tracing JIT constrains
the trajectory), it is built by **recursive doubling** -- the natural,
checkpoint-free formulation. The recursion depth is bounded by ``max_treedepth``,
and chains that U-turn or diverge early are *masked off* so the remaining chains
keep doubling; the outer loop stops once every chain has terminated.

It shares ``hmc_mlx``'s conventions exactly and reuses its helpers:

* The position ``q`` may be any MLX **pytree**; the leapfrog and U-turn checks
  are written with ``mlx.utils.tree_map``, so a plain array is the single-leaf
  case.
* The ``logp_dlogp_func`` is a single-chain, MLX-traceable ``q -> (logp, dlogp)``,
  batched over chains internally with ``mx.vmap``.
* Only **diagonal** mass matrices are supported (``velocity = var * p``,
  ``p ~ N(0, 1/var)``).

Divergence handling: a leaf is flagged divergent when its energy change is
non-finite or exceeds ``Emax``; divergent or turning subtrees are never selected
as the next proposal.
"""

import mlx.core as mx
from mlx.utils import tree_map

from .hmc_mlx import (
    _adapt_and_sample,
    _bcast,
    _init_start,
    _kinetic_energy,
    _tree_dot,
    _tree_random_momentum,
    _where_tree,
)


def _tree_add(a, b):
    return tree_map(mx.add, a, b)


def _tree_sub(a, b):
    return tree_map(mx.subtract, a, b)


def _velocity(p, var):
    """Diagonal velocity ``var * p`` (event-shaped ``var`` broadcasts over chains)."""
    return tree_map(mx.multiply, p, var)


class _Tree:
    """A (sub)tree of the NUTS trajectory, batched over chains on the leading axis.

    Position/momentum fields (``z_*``, ``r_*``, ``grad_*``, ``z_proposal``,
    ``r_sum``) are pytrees with ``(chains, *event)`` leaves; the running
    statistics (``weight``, ``turning``, ``diverging``, ``sum_accept_probs``,
    ``num_proposals``) are ``(chains,)`` arrays. Mirrors NumPyro's ``TreeInfo``.
    """

    __slots__ = (
        "z_left", "r_left", "grad_left",
        "z_right", "r_right", "grad_right",
        "z_proposal", "weight", "r_sum",
        "turning", "diverging", "sum_accept_probs", "num_proposals",
    )

    def __init__(self, z_left, r_left, grad_left, z_right, r_right, grad_right,
                 z_proposal, weight, r_sum, turning, diverging,
                 sum_accept_probs, num_proposals):
        self.z_left, self.r_left, self.grad_left = z_left, r_left, grad_left
        self.z_right, self.r_right, self.grad_right = z_right, r_right, grad_right
        self.z_proposal, self.weight, self.r_sum = z_proposal, weight, r_sum
        self.turning, self.diverging = turning, diverging
        self.sum_accept_probs, self.num_proposals = sum_accept_probs, num_proposals


def _leapfrog(batched_ldf, z, r, grad, signed_step, var):
    """One velocity-Verlet leapfrog step over all chains, leafwise via ``tree_map``.

    ``signed_step`` is a ``(chains,)`` array carrying each chain's direction
    (positive to extend right, negative to extend left).
    """
    half = 0.5 * signed_step
    r = tree_map(lambda rr, g: rr + _bcast(half, rr) * g, r, grad)
    z = tree_map(lambda zz, rr, v: zz + _bcast(signed_step, zz) * (v * rr), z, r, var)
    logp, grad = batched_ldf(z)
    r = tree_map(lambda rr, g: rr + _bcast(half, rr) * g, r, grad)
    return z, r, grad, logp


def _base_tree(batched_ldf, z, r, grad, var, signed_step, energy0, Emax):
    """Take one leapfrog step and wrap the new state as a depth-0 ``_Tree``."""
    z, r, grad, logp = _leapfrog(batched_ldf, z, r, grad, signed_step, var)
    energy = _kinetic_energy(r, var) - logp
    delta = energy - energy0  # (chains,)
    delta = mx.where(mx.isnan(delta), mx.inf, delta)
    weight = -delta
    diverging = delta > Emax
    accept_prob = mx.minimum(1.0, mx.exp(-delta))
    ones = mx.ones_like(accept_prob)
    return _Tree(
        z, r, grad, z, r, grad, z,
        weight, r,
        mx.zeros_like(accept_prob) > 0,  # turning = False (chains,)
        diverging, accept_prob, ones,
    )


def _is_turning(r_left, r_right, r_sum, var):
    """Generalized U-turn criterion (Betancourt 2017), per chain -> ``(chains,)``."""
    r_mid = tree_map(lambda x: 0.5 * x, _tree_add(r_left, r_right))
    r_adj = _tree_sub(r_sum, r_mid)
    left_angle = _tree_dot(_velocity(r_left, var), r_adj)
    right_angle = _tree_dot(_velocity(r_right, var), r_adj)
    return (left_angle <= 0) | (right_angle <= 0)


def _edge(tree, going_right):
    """The boundary integrator state of ``tree`` in each chain's direction."""
    z = _where_tree(going_right, tree.z_right, tree.z_left)
    r = _where_tree(going_right, tree.r_right, tree.r_left)
    grad = _where_tree(going_right, tree.grad_right, tree.grad_left)
    return z, r, grad


def _combine(cur, new, var, going_right, key, biased):
    """Merge ``new`` into ``cur`` in each chain's doubling direction.

    Outer boundaries come from whichever subtree is on each side. The proposal is
    resampled with a biased kernel between main trees (favouring the fresh half)
    and a uniform (multinomial) kernel within a subtree, matching NumPyro. The
    combined ``turning`` flag includes the cross-subtree generalized U-turn.
    """
    z_left, r_left, grad_left = _where_tree(
        going_right, cur.z_left, new.z_left), _where_tree(
        going_right, cur.r_left, new.r_left), _where_tree(
        going_right, cur.grad_left, new.grad_left)
    z_right, r_right, grad_right = _where_tree(
        going_right, new.z_right, cur.z_right), _where_tree(
        going_right, new.r_right, cur.r_right), _where_tree(
        going_right, new.grad_right, cur.grad_right)

    r_sum = _tree_add(cur.r_sum, new.r_sum)
    weight = mx.logaddexp(cur.weight, new.weight)
    cross_turning = _is_turning(r_left, r_right, r_sum, var)
    turning = cur.turning | new.turning | cross_turning
    diverging = cur.diverging | new.diverging

    if biased:
        # Favour moving onto the freshly built half (Betancourt progressive bias).
        transition_prob = mx.minimum(1.0, mx.exp(new.weight - cur.weight))
        transition_prob = mx.where(new.turning | new.diverging, 0.0, transition_prob)
    else:
        transition_prob = mx.sigmoid(new.weight - cur.weight)

    accept = mx.random.bernoulli(transition_prob, key=key)
    z_proposal = _where_tree(accept, new.z_proposal, cur.z_proposal)

    return _Tree(
        z_left, r_left, grad_left, z_right, r_right, grad_right,
        z_proposal, weight, r_sum, turning, diverging,
        cur.sum_accept_probs + new.sum_accept_probs,
        cur.num_proposals + new.num_proposals,
    )


def _build_subtree(batched_ldf, z, r, grad, var, signed_step, going_right,
                   depth, energy0, Emax, key):
    """Recursively build a balanced subtree of ``2 ** depth`` leaves from one edge.

    Depth 0 is a single leapfrog leaf; deeper trees combine two same-depth halves
    built in sequence (the second from the far edge of the first), checking the
    generalized U-turn as they merge. All chains are built to full depth; the
    per-chain ``turning`` / ``diverging`` flags carry which ones should stop.
    """
    if depth == 0:
        return _base_tree(batched_ldf, z, r, grad, var, signed_step, energy0, Emax)

    k_left, k_right, k_comb = mx.random.split(key, 3)
    left = _build_subtree(
        batched_ldf, z, r, grad, var, signed_step, going_right,
        depth - 1, energy0, Emax, k_left,
    )
    z2, r2, grad2 = _edge(left, going_right)
    right = _build_subtree(
        batched_ldf, z2, r2, grad2, var, signed_step, going_right,
        depth - 1, energy0, Emax, k_right,
    )
    return _combine(left, right, var, going_right, k_comb, biased=False)


def _build_nuts_move(batched_ldf, max_treedepth, Emax):
    """Build a single batched NUTS transition over all chains.

    Returns ``move(q, key, step_size, var) -> (q_next, accept_prob, diverging)``
    matching ``hmc_mlx``'s move, so the shared dual-averaging driver advances it.
    """

    def move(q, key, step_size, var):
        key_p, key = mx.random.split(key)
        logp0, grad0 = batched_ldf(q)
        r0 = _tree_random_momentum(key_p, q, var)
        energy0 = _kinetic_energy(r0, var) - logp0  # (chains,)
        n_chains = energy0.shape[0]
        false = mx.zeros_like(energy0) > 0

        tree = _Tree(
            q, r0, grad0, q, r0, grad0, q,
            mx.zeros_like(energy0), r0, false, false,
            mx.zeros_like(energy0), mx.zeros_like(energy0),
        )
        active = mx.ones_like(energy0) > 0  # (chains,) bool

        for depth in range(int(max_treedepth)):
            key, k_dir, k_build, k_comb = mx.random.split(key, 4)
            going_right = mx.random.bernoulli(0.5, shape=(n_chains,), key=k_dir)
            signed_step = mx.where(going_right, step_size, -step_size)
            z, r, grad = _edge(tree, going_right)
            new_sub = _build_subtree(
                batched_ldf, z, r, grad, var, signed_step, going_right,
                depth, energy0, Emax, k_build,
            )
            merged = _combine(tree, new_sub, var, going_right, k_comb, biased=True)
            # Freeze chains that already terminated in an earlier doubling.
            tree = _freeze(active, merged, tree)
            active = active & ~(new_sub.turning | new_sub.diverging | merged.turning)
            if not mx.any(active).item():
                break

        accept_prob = tree.sum_accept_probs / tree.num_proposals
        return tree.z_proposal, accept_prob, tree.diverging

    return move


def _freeze(active, new_tree, old_tree):
    """Per-chain select ``new_tree`` where ``active`` else keep ``old_tree``.

    Only the fields needed downstream (the proposal and the running statistics)
    are selected; the boundary states are taken from ``new_tree`` since inactive
    chains never extend again."""
    z_proposal = _where_tree(active, new_tree.z_proposal, old_tree.z_proposal)
    pick = lambda a, b: mx.where(active, a, b)
    return _Tree(
        new_tree.z_left, new_tree.r_left, new_tree.grad_left,
        new_tree.z_right, new_tree.r_right, new_tree.grad_right,
        z_proposal, new_tree.weight, new_tree.r_sum,
        pick(new_tree.turning, old_tree.turning),
        pick(new_tree.diverging, old_tree.diverging),
        pick(new_tree.sum_accept_probs, old_tree.sum_accept_probs),
        pick(new_tree.num_proposals, old_tree.num_proposals),
    )


def sample_nuts_chains(
    logp_dlogp_func,
    model_ndim,
    draws=1000,
    tune=1000,
    chains=4,
    max_treedepth=10,
    target_accept=0.8,
    init_step=0.1,
    Emax=1000.0,
    adapt_mass=True,
    random_seed=0,
    start=None,
):
    """Sample multiple chains with a batched, on-device NUTS (MLX).

    All ``chains`` advance together as the leading array axis. Each draw builds an
    adaptive No-U-Turn trajectory (recursive doubling, generalized U-turn
    termination) instead of a fixed leapfrog path. Warmup tunes a single (shared)
    step size by dual averaging and, if ``adapt_mass`` is True, a diagonal mass
    matrix estimated from the warmup draws; both are then frozen for sampling.

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
    max_treedepth : int
        Maximum NUTS tree depth (a trajectory has at most ``2 ** max_treedepth``
        leapfrog steps before being forced to stop).
    target_accept : float
        Target mean acceptance for dual-averaging step-size adaptation.
    init_step : float
        Initial leapfrog step size before adaptation.
    Emax : float
        Energy-change threshold above which a leaf is flagged divergent.
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
    move = _build_nuts_move(batched_ldf, int(max_treedepth), Emax)
    return _adapt_and_sample(
        move, target_accept, q0, key, model_ndim, draws, tune, init_step, adapt_mass
    )
