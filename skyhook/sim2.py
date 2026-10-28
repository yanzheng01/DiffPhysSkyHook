"""Host-layer simulation for the semi-active quarter-car (model II + ON-OFF).

run_semi: single-kernel differentiable rollout (model II force-domain clip);
the same launch sequence (set params -> set IC -> rollout_semi -> reduce ->
finalize) is what the AD tests wrap in qd.ad.Tape / qd.ad.FwdMode.

run_onoff: ON-OFF baseline, launched through a qd.ad.grad_replaced wrapper
whose custom gradient is identically zero. Outside a Tape/FwdMode the wrapper
is a plain forward call; inside a Tape the ON-OFF rollout contributes exact
zero gradients (the switching law is discontinuous -- a loud structural zero
instead of silently differentiating through the qd.select branch).

run_semi_cmd (plan B): command-driven rollout -- the damper force command
comes from the external series f_cmd[n] (step-wise ZOH) instead of the
ideal skyhook law; realizability is enforced model-side by fd_clip_cmd.
"""

from dataclasses import dataclass

import numpy as np
import quadrants as qd

from . import model2
from .params import SemiActiveParams


@dataclass
class SemiResult:
    """Outputs of one run (channels indexed at whole steps t_n, n = 0..N-1)."""

    t: np.ndarray           # (N,)   t_n
    state: np.ndarray       # (N, 4) x1, x2, v1, v2 at t_n
    a2: np.ndarray          # (N,)   v2dot at t_n (J_acc channel, incl. F_d)
    fd: np.ndarray          # (N,)   device force F_d at t_n
    mask: np.ndarray        # (N,)   clip activation flag (1 = F_sky clipped)
    ft: np.ndarray          # (N,)   tire force mu*omega_t^2*(x1 - z_r) at t_n
    ct: np.ndarray = None   # (N,)   c(t) (ON-OFF branch only)
    final_state: tuple = None  # state at t_N


def _prepare(params: SemiActiveParams, road, x0, t_end: float, dt: float):
    n_steps = int(round(t_end / dt))
    if n_steps < 1:
        raise ValueError(f"t_end={t_end} too short for dt={dt}")
    if n_steps > model2.T_MAX:
        raise ValueError(f"n_steps={n_steps} exceeds T_MAX={model2.T_MAX}")
    model2.set_params2(params.mu, params.omega_t_sq, params.omega_s_sq,
                       params.gamma_t, params.gamma_sky, params.c_min,
                       params.c_max)
    model2.set_state_hist2(*x0)
    kind, amp, w, t0, width, ramp_t = road.kernel_args()
    return n_steps, (kind, amp, w, t0, width, ramp_t)


def _collect(n_steps: int, dt: float, with_ct: bool) -> SemiResult:
    state = model2.traj2.to_numpy()[:n_steps]
    hist = model2.x_hist2.to_numpy()
    return SemiResult(
        t=np.arange(n_steps, dtype=np.float64) * dt,
        state=state,
        a2=model2.acc_rec2.to_numpy()[:n_steps],
        fd=model2.fd_rec2.to_numpy()[:n_steps],
        mask=model2.mask_rec2.to_numpy()[:n_steps],
        ft=model2.ft_rec2.to_numpy()[:n_steps],
        ct=model2.ct_rec2.to_numpy()[:n_steps] if with_ct else None,
        final_state=tuple(hist[n_steps]),
    )


def run_semi(params: SemiActiveParams, road, x0=(0.0, 0.0, 0.0, 0.0),
             t_end: float = 2.0, dt: float = 1.0e-3) -> SemiResult:
    """Forward rollout of model II (force-domain clip); differentiable path.

    For gradients wrap the same launch sequence (set_params2 / set_state_hist2
    / rollout_semi / reduce_j2 / finalize_j2) in qd.ad.Tape or qd.ad.FwdMode,
    as in tests/test_d3_semi.py (T13); this helper is the plain forward API.
    """
    n_steps, road_args = _prepare(params, road, x0, t_end, dt)
    model2.rollout_semi(n_steps, dt, *road_args)
    return _collect(n_steps, dt, with_ct=False)


def run_semi_cmd(params: SemiActiveParams, f_cmd, road,
                 x0=(0.0, 0.0, 0.0, 0.0),
                 t_end: float = 2.0, dt: float = 1.0e-3) -> SemiResult:
    """Forward rollout of model II driven by an external force-command series.

    f_cmd[n] is the desired damper force for step n (t_n -> t_{n+1}), held
    constant across the RK4 stages of that step (ZOH, plan B section 2-1).
    Realizability is enforced model-side by fd_clip_cmd (no optimizer-side
    projection); res.mask reports the command out-of-band rate. The command
    is written element-by-element before the rollout launch (one kernel per
    sample -- known overhead, plan B section 6).

    For gradients wrap the same launch sequence (set_params2 / set_state_hist2
    / set_fcmd_elem / rollout_cmd_semi / reduce_j2 / finalize_j2) in
    qd.ad.Tape (reverse: whole (T,) adjoint in one backward pass) or
    qd.ad.FwdMode (JVP spot checks), as in tests/test_e2_force_grad.py.
    """
    n_steps, road_args = _prepare(params, road, x0, t_end, dt)
    fc = np.asarray(f_cmd, dtype=np.float64)
    if fc.shape != (n_steps,):
        raise ValueError(f"f_cmd must have shape ({n_steps},), got {fc.shape}")
    for n in range(n_steps):
        model2.set_fcmd_elem(n, float(fc[n]))
    model2.rollout_cmd_semi(n_steps, dt, *road_args)
    return _collect(n_steps, dt, with_ct=False)


@qd.ad.grad_replaced
def _onoff_rollout(n_steps, dt, kind, amp, w, t0, width, ramp_t):
    model2.rollout_onoff(n_steps, dt, kind, amp, w, t0, width, ramp_t)


@qd.ad.grad_for(_onoff_rollout)
def _onoff_zero_grad(n_steps, dt, kind, amp, w, t0, width, ramp_t):
    """ON-OFF switching law is discontinuous: its gradient is defined as zero."""
    pass


def run_onoff(params: SemiActiveParams, road, x0=(0.0, 0.0, 0.0, 0.0),
              t_end: float = 2.0, dt: float = 1.0e-3) -> SemiResult:
    """Forward-only ON-OFF baseline rollout (section 4.3), zero-gradient marked."""
    n_steps, road_args = _prepare(params, road, x0, t_end, dt)
    _onoff_rollout(n_steps, dt, *road_args)
    return _collect(n_steps, dt, with_ct=True)
