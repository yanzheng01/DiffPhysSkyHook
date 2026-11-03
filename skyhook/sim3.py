"""Host layer for model3 (plan C, F0): discrete-road command-driven rollouts.

run_cmd_seq_batch: batched rollout of B command-driven model-II trajectories
on DISCRETE road sequences (skyhook/road_seq.py), with the composite MS loss
J_sim (plan C decisions 3/8). The launch sequence mirrors sim2b exactly:

    _setup + _setup_road + _setup_fcmd   (all OUTSIDE any Tape/FwdMode --
                                          a parameter store inside the
                                          context would sever the adjoint
                                          chain, D3 pattern)
    -> run_jsim_chain                    (rollout + reductions, the
                                          AD-wrappable part)

Segment interface (plan C decision 1): i0 is the global half-step index of
the segment start; road_seqs cover the WHOLE episode grid and are written
once, each segment rollout reads them with its offset. f_cmds / x0s /
n_steps are per SEGMENT (t_end = H*dt of the segment).

The four sum-of-squares accumulators stay readable after the run, so
j_terms[b] = (MS a2, MS travel, MS tire, MS hinge) needs no extra launch.
"""

from dataclasses import dataclass

import numpy as np

from . import model3
from .params import SemiActiveParams
from .road_seq import RoadSeq

# default loss weights (parameters_reference.md section 4.1: initial guess,
# to be tuned in the F2 smoke runs) and default stroke envelope (decision 8)
DEFAULT_WEIGHTS = (1.0, 0.1, 0.1, 0.5)
X_MAX_DEFAULT = 0.08


@dataclass
class SeqRunResult:
    """Outputs of one batched discrete-road run (whole-step records)."""

    t: np.ndarray            # (N,)    segment t_n grid (local time)
    state: np.ndarray        # (B, N, 4) x1, x2, v1, v2 at t_n
    a2: np.ndarray           # (B, N)  v2dot at t_n (J_acc channel, incl. F_d)
    fd: np.ndarray           # (B, N)  device force F_d at t_n
    mask: np.ndarray         # (B, N)  COMMAND out-of-band flag
    ft: np.ndarray           # (B, N)  tire force mu*omega_t^2*(x1 - z_r) at t_n
    travel: np.ndarray       # (B, N)  suspension stroke x2 - x1 at t_n
    j_sim: np.ndarray        # (B,)    composite MS loss per batch element
    j_terms: np.ndarray      # (B, 4)  MS terms (acc, disp, tire, hinge)
    j_sim_mean: float        # mean_b j_sim[b] (the scalar loss for AD)
    final_state: np.ndarray  # (B, 4)  state at t_N


def _setup(params_list, x0s, xmaxs) -> int:
    """Write params / envelopes / ICs; returns B (validates shapes)."""
    B = len(params_list)
    if B < 1:
        raise ValueError("params_list must be non-empty")
    if len(x0s) != B:
        raise ValueError(f"len(x0s)={len(x0s)} != B={B}")
    if len(xmaxs) != B:
        raise ValueError(f"len(xmaxs)={len(xmaxs)} != B={B}")
    if B > model3.B_MAX:
        raise ValueError(f"batch size {B} exceeds B_MAX={model3.B_MAX}")

    p0 = params_list[0]
    model3.set_shared_params3(p0.mu, p0.omega_t_sq, p0.omega_s_sq,
                              p0.gamma_t)
    for b in range(B):
        p = params_list[b]
        model3.set_params3(b, p.gamma_sky, p.c_min, p.c_max)
        model3.set_xmax3(b, xmaxs[b])
        model3.set_state_hist3(b, *x0s[b])
    return B


def _setup_road(road_seqs, i0: int, n_steps: int, B: int) -> None:
    """Write the half-step sequences element-wise; validate grid coverage.

    Each sequence must cover the rollout span: max grid index read is
    i0 + 2*n_steps (stage 4 of the last step), so the sequence needs
    i0 + 2*n_steps <= m - 1. Sequences may be longer than the span (a
    full-episode grid reused by every segment rollout).
    """
    if len(road_seqs) != B:
        raise ValueError(f"len(road_seqs)={len(road_seqs)} != B={B}")
    if i0 < 0 or i0 % 2 != 0:
        raise ValueError(f"i0 must be a non-negative even half-step index, "
                         f"got {i0}")
    for b in range(B):
        seq = road_seqs[b]
        if not isinstance(seq, RoadSeq):
            raise ValueError(f"road_seqs[{b}] is not a RoadSeq")
        if seq.m > model3.RS_MAX:
            raise ValueError(f"road_seqs[{b}].m={seq.m} exceeds RS_MAX="
                             f"{model3.RS_MAX}")
        if i0 + 2 * n_steps > seq.m - 1:
            raise ValueError(
                f"road_seqs[{b}] covers {seq.m} half-steps; rollout needs "
                f"i0+2*n_steps+1 = {i0 + 2 * n_steps + 1}")
        for i in range(seq.m):
            model3.set_road_seq_elem(b, i, float(seq.z[i]), float(seq.zd[i]))


def _setup_fcmd(f_cmds, n_steps: int, B: int) -> None:
    """Write the (B, n_steps) command block element-wise (E4 pattern)."""
    fc = np.asarray(f_cmds, dtype=np.float64)
    if fc.shape != (B, n_steps):
        raise ValueError(f"f_cmds must have shape ({B}, {n_steps}), "
                         f"got {fc.shape}")
    for b in range(B):
        for n in range(n_steps):
            model3.set_fcmd3_elem(b, n, float(fc[b, n]))


def run_jsim_chain(n_steps: int, dt: float, B: int, i0: int = 0,
                   weights=DEFAULT_WEIGHTS) -> None:
    """Rollout + J_sim reductions: the AD-wrappable launch sequence.

    Setup (_setup + _setup_road + _setup_fcmd) must run OUTSIDE any
    Tape/FwdMode context (D3 pattern).
    """
    w_acc, w_disp, w_tire, w_lim = weights
    model3.rollout_cmd_seq_batch(n_steps, dt, B, i0)
    model3.clear_jsim3()
    model3.reduce_jsim3(n_steps, B)
    model3.finalize_jsim3(n_steps, B, w_acc, w_disp, w_tire, w_lim)
    model3.clear_mean_jsim3()
    model3.reduce_mean_jsim3(B)


def run_cmd_seq_batch(params_list, f_cmds, road_seqs, x0s=None, xmaxs=None,
                      weights=DEFAULT_WEIGHTS, t_end: float = 1.0,
                      dt: float = 1.0e-3, i0: int = 0) -> SeqRunResult:
    """Batched command-driven rollout on discrete road sequences.

    f_cmds[b, n] is the desired damper force of element b at (segment)
    step n (ZOH across the RK4 stages); realizability is enforced
    model-side by fd_clip_cmd3 (no optimizer-side projection). x0s
    defaults to rest ICs -- Plan C training must pass NONZERO ICs or rely
    on the nonzero road (discipline, parameters_reference.md section 4.4).
    xmaxs defaults to X_MAX_DEFAULT for every element.
    """
    n_steps = int(round(t_end / dt))
    if n_steps < 1:
        raise ValueError(f"t_end={t_end} too short for dt={dt}")
    if n_steps > model3.T_B_MAX - 1:
        raise ValueError(f"n_steps={n_steps} exceeds T_B_MAX-1="
                         f"{model3.T_B_MAX - 1}")
    B = len(params_list)
    if x0s is None:
        x0s = [(0.0, 0.0, 0.0, 0.0)] * B
    if xmaxs is None:
        xmaxs = [X_MAX_DEFAULT] * B

    _setup(params_list, x0s, xmaxs)
    _setup_road(road_seqs, i0, n_steps, B)
    _setup_fcmd(f_cmds, n_steps, B)
    run_jsim_chain(n_steps, dt, B, i0, weights)

    j_terms = np.stack([
        model3.js_acc3.to_numpy()[:B] / n_steps,
        model3.js_disp3.to_numpy()[:B] / n_steps,
        model3.js_tire3.to_numpy()[:B] / n_steps,
        model3.js_lim3.to_numpy()[:B] / n_steps,
    ], axis=1)
    return SeqRunResult(
        t=np.arange(n_steps, dtype=np.float64) * dt,
        state=model3.traj3.to_numpy()[:B, :n_steps],
        a2=model3.acc3.to_numpy()[:B, :n_steps],
        fd=model3.fd3.to_numpy()[:B, :n_steps],
        mask=model3.mask3.to_numpy()[:B, :n_steps],
        ft=model3.ft3.to_numpy()[:B, :n_steps],
        travel=model3.travel3.to_numpy()[:B, :n_steps],
        j_sim=model3.j_val3.to_numpy()[:B],
        j_terms=j_terms,
        j_sim_mean=model3.j_mean3[None],
        final_state=model3.xh3.to_numpy()[:B, n_steps],
    )
