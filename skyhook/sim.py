"""Host-layer simulation loop for the passive quarter-car (RK4, dt = 1 ms)."""

from dataclasses import dataclass

import numpy as np

from . import model
from .params import PassiveParams


@dataclass
class SimResult:
    """Outputs of one run (channels indexed at whole steps t_n, n = 0..N-1)."""

    t: np.ndarray          # (N,)   t_n
    state: np.ndarray      # (N, 4) x1, x2, v1, v2 at t_n
    a2: np.ndarray         # (N,)   v2dot at t_n (J_acc channel)
    fd: np.ndarray         # (N,)   suspension damper force F_d at t_n
    final_state: tuple     # state at t_N


def run_passive(params: PassiveParams, road, x0=(0.0, 0.0, 0.0, 0.0),
                t_end: float = 2.0, dt: float = 1.0e-3) -> SimResult:
    """Integrate the passive quarter-car from x0 over [0, t_end].

    The time loop stays in the Python host (architecture rule); each step is
    one loop-free kernel launch. Road input is analytic and gradient-detached.
    """
    n_steps = int(round(t_end / dt))
    if n_steps < 1:
        raise ValueError(f"t_end={t_end} too short for dt={dt}")
    if n_steps > model.T_MAX:
        raise ValueError(f"n_steps={n_steps} exceeds T_MAX={model.T_MAX}")

    model.set_params(params.mu, params.omega_t_sq, params.omega_s_sq,
                     params.gamma_t, params.gamma_s, params.gamma_sky)
    model.set_state(*x0)

    kind, amp, w, t0, width, ramp_t = road.kernel_args()
    for n in range(n_steps):
        model.rk4_step(n * dt, dt, n, kind, amp, w, t0, width, ramp_t)

    state = model.traj.to_numpy()[:n_steps]
    a2 = model.acc_rec.to_numpy()[:n_steps]
    fd = model.fd_rec.to_numpy()[:n_steps]
    final_state = (model.x1[None], model.x2[None],
                   model.v1[None], model.v2[None])
    t = np.arange(n_steps, dtype=np.float64) * dt
    return SimResult(t=t, state=state, a2=a2, fd=fd, final_state=final_state)


def run_passive_euler(params: PassiveParams, road, x0=(0.0, 0.0, 0.0, 0.0),
                      t_end: float = 2.0, dt: float = 1.0e-3) -> SimResult:
    """Same host loop as run_passive but with the semi-implicit Euler step.

    Convergence-order companion for T10 only (plan rule: RK4 at dt = 1 ms is
    the production integrator; Euler exists to measure the observed order).
    """
    n_steps = int(round(t_end / dt))
    if n_steps < 1:
        raise ValueError(f"t_end={t_end} too short for dt={dt}")
    if n_steps > model.T_MAX:
        raise ValueError(f"n_steps={n_steps} exceeds T_MAX={model.T_MAX}")

    model.set_params(params.mu, params.omega_t_sq, params.omega_s_sq,
                     params.gamma_t, params.gamma_s, params.gamma_sky)
    model.set_state(*x0)

    kind, amp, w, t0, width, ramp_t = road.kernel_args()
    for n in range(n_steps):
        model.euler_step(n * dt, dt, n, kind, amp, w, t0, width, ramp_t)

    state = model.traj.to_numpy()[:n_steps]
    a2 = model.acc_rec.to_numpy()[:n_steps]
    fd = model.fd_rec.to_numpy()[:n_steps]
    final_state = (model.x1[None], model.x2[None],
                   model.v1[None], model.v2[None])
    t = np.arange(n_steps, dtype=np.float64) * dt
    return SimResult(t=t, state=state, a2=a2, fd=fd, final_state=final_state)
