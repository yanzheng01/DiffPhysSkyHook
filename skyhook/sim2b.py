"""Host layer for the batched semi-active model (plan A, D4, model2b).

run_semi_batch: batched forward rollout of B independent model-II
trajectories -- per-batch tunable trio (gamma_sky, c_min, c_max), per-batch
roads, shared hardware gains and time grid. The same launch sequence
(set_shared_params_b / set_params_b / set_road_b / set_state_hist_b ->
rollout_semi_batch -> clear_j_b / reduce_j_b / finalize_j_b ->
clear_mean_j_b / reduce_mean_j_b) is what the batched AD tests wrap in
qd.ad.Tape / qd.ad.FwdMode (D4 probe: reverse gives the full gradient
vector on pg_b/pcmin_b/pcmax_b; FwdMode seeds an element of the (B_MAX,)
param field via a full-length seed list).

run_semi_cmd_batch (plan B, E4): command-driven batched rollout -- the
device force command comes from the external block f_cmds[b, n] (step-wise
ZOH) instead of the ideal skyhook law; realizability is enforced model-side
by model2b.fd_clip_cmd_b, and res.mask reports the COMMAND out-of-band rate
on that rollout. The AD-wrappable sequence is (_setup + _setup_fcmd) outside
any Tape, then run_cmd_j_chain inside.

Numerical caveat (D4 probe finding): j_val_b[b] = sqrt(j_acc_sq_b[b]/n) has
dual sqrt'(0) = 0/0 = nan, so a batch element with J = 0 (flat road + zero IC)
poisons the dual chain of j_mean. Give flat-road elements a nonzero IC
whenever the run feeds FwdMode.
"""

from dataclasses import dataclass

import numpy as np

from . import model2b
from .params import SemiActiveParams


@dataclass
class BatchSemiResult:
    """Outputs of one batched run (whole-step records, n = 0..N-1)."""

    t: np.ndarray            # (N,)    shared t_n grid
    state: np.ndarray        # (B, N, 4) x1, x2, v1, v2 at t_n
    a2: np.ndarray           # (B, N)  v2dot at t_n (J_acc channel, incl. F_d)
    fd: np.ndarray           # (B, N)  device force F_d at t_n
    mask: np.ndarray         # (B, N)  clip activation flag (1 = F_sky clipped)
    ft: np.ndarray           # (B, N)  tire force mu*omega_t^2*(x1 - z_r) at t_n
    j: np.ndarray            # (B,)    RMS(acc) per batch element
    j_mean: float            # mean_b j[b] (the scalar loss for AD)
    final_state: np.ndarray  # (B, 4)  state at t_N


def _setup(params_list, roads, x0s) -> int:
    """Write all batch input fields; returns B (validates shapes)."""
    B = len(params_list)
    if B < 1:
        raise ValueError("params_list must be non-empty")
    if len(roads) != B:
        raise ValueError(f"len(roads)={len(roads)} != len(params_list)={B}")
    if B > model2b.B_MAX:
        raise ValueError(f"batch size {B} exceeds B_MAX={model2b.B_MAX}")
    if len(x0s) != B:
        raise ValueError(f"len(x0s)={len(x0s)} != B={B}")

    p0 = params_list[0]
    model2b.set_shared_params_b(p0.mu, p0.omega_t_sq, p0.omega_s_sq,
                                p0.gamma_t)
    for b in range(B):
        p = params_list[b]
        model2b.set_params_b(b, p.gamma_sky, p.c_min, p.c_max)
        model2b.set_road_b(b, *roads[b].kernel_args())
        model2b.set_state_hist_b(b, *x0s[b])
    return B


def run_j_chain(n_steps: int, dt: float, B: int):
    """Rollout + J reductions: the AD-wrappable launch sequence.

    Setup (_setup) must run OUTSIDE any Tape/FwdMode context (a parameter
    store inside the context would sever the adjoint chain, D3 pattern).
    """
    model2b.rollout_semi_batch(n_steps, dt, B)
    model2b.clear_j_b()
    model2b.reduce_j_b(n_steps, B)
    model2b.finalize_j_b(n_steps, B)
    model2b.clear_mean_j_b()
    model2b.reduce_mean_j_b(B)


def run_semi_batch(params_list, roads, x0s=None,
                   t_end: float = 2.0, dt: float = 1.0e-3) -> BatchSemiResult:
    """Batched forward rollout of model II; B = len(params_list) <= B_MAX.

    The shared hardware gains (mu, omega_t^2, omega_s^2, gamma_t) are taken
    from params_list[0]; only the tunable trio is per-batch. x0s defaults to
    zero initial states (see the module docstring for the J = 0 dual caveat).
    """
    n_steps = int(round(t_end / dt))
    if n_steps < 1:
        raise ValueError(f"t_end={t_end} too short for dt={dt}")
    if n_steps > model2b.T_B_MAX - 1:
        raise ValueError(f"n_steps={n_steps} exceeds T_B_MAX-1="
                         f"{model2b.T_B_MAX - 1}")
    if x0s is None:
        x0s = [(0.0, 0.0, 0.0, 0.0)] * len(params_list)

    B = _setup(params_list, roads, x0s)
    run_j_chain(n_steps, dt, B)

    return BatchSemiResult(
        t=np.arange(n_steps, dtype=np.float64) * dt,
        state=model2b.traj_b.to_numpy()[:B, :n_steps],
        a2=model2b.acc_b.to_numpy()[:B, :n_steps],
        fd=model2b.fd_b.to_numpy()[:B, :n_steps],
        mask=model2b.mask_b.to_numpy()[:B, :n_steps],
        ft=model2b.ft_b.to_numpy()[:B, :n_steps],
        j=model2b.j_val_b.to_numpy()[:B],
        j_mean=model2b.j_mean[None],
        final_state=model2b.xh_b.to_numpy()[:B, n_steps],
    )


def _setup_fcmd(f_cmds, n_steps: int, B: int) -> None:
    """Write the (B, n_steps) command block element-wise (plan B section 6).

    Must run OUTSIDE any Tape/FwdMode context (same reasoning as _setup):
    a parameter store inside the context would sever the adjoint chain.
    Element-wise launches (B x n_steps per setup) are a known, accepted
    overhead for the smoke scale (plan B section 6).
    """
    fc = np.asarray(f_cmds, dtype=np.float64)
    if fc.shape != (B, n_steps):
        raise ValueError(f"f_cmds must have shape ({B}, {n_steps}), "
                         f"got {fc.shape}")
    for b in range(B):
        for n in range(n_steps):
            model2b.set_fcmd_batch_elem(b, n, float(fc[b, n]))


def run_cmd_j_chain(n_steps: int, dt: float, B: int):
    """Command-driven rollout + J reductions: the AD-wrappable launch sequence.

    Setup (_setup + _setup_fcmd) must run OUTSIDE any Tape/FwdMode context.
    On this rollout model2b.mask_b is the COMMAND out-of-band channel
    (clip_mask_cmd_b), not the skyhook-law clip mask.
    """
    model2b.rollout_semi_cmd_batch(n_steps, dt, B)
    model2b.clear_j_b()
    model2b.reduce_j_b(n_steps, B)
    model2b.finalize_j_b(n_steps, B)
    model2b.clear_mean_j_b()
    model2b.reduce_mean_j_b(B)


def run_semi_cmd_batch(params_list, f_cmds, roads, x0s=None,
                       t_end: float = 2.0, dt: float = 1.0e-3) -> BatchSemiResult:
    """Batched forward rollout of model II driven by external force commands.

    f_cmds[b, n] is the desired damper force of element b at step n (t_n ->
    t_{n+1}), held constant across the RK4 stages of that step (ZOH, plan B
    section 2-1). Realizability is enforced model-side by fd_clip_cmd_b (no
    optimizer-side projection); res.mask reports the COMMAND out-of-band
    rate (not the skyhook-law clip mask of run_semi_batch).
    """
    n_steps = int(round(t_end / dt))
    if n_steps < 1:
        raise ValueError(f"t_end={t_end} too short for dt={dt}")
    if n_steps > model2b.T_B_MAX - 1:
        raise ValueError(f"n_steps={n_steps} exceeds T_B_MAX-1="
                         f"{model2b.T_B_MAX - 1}")
    if x0s is None:
        x0s = [(0.0, 0.0, 0.0, 0.0)] * len(params_list)

    B = _setup(params_list, roads, x0s)
    _setup_fcmd(f_cmds, n_steps, B)
    run_cmd_j_chain(n_steps, dt, B)

    return BatchSemiResult(
        t=np.arange(n_steps, dtype=np.float64) * dt,
        state=model2b.traj_b.to_numpy()[:B, :n_steps],
        a2=model2b.acc_b.to_numpy()[:B, :n_steps],
        fd=model2b.fd_b.to_numpy()[:B, :n_steps],
        mask=model2b.mask_b.to_numpy()[:B, :n_steps],
        ft=model2b.ft_b.to_numpy()[:B, :n_steps],
        j=model2b.j_val_b.to_numpy()[:B],
        j_mean=model2b.j_mean[None],
        final_state=model2b.xh_b.to_numpy()[:B, n_steps],
    )
