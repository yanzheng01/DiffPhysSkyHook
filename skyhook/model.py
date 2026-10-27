"""Quadrants kernels for the passive quarter-car (plan A, D1: gamma_sky = 0).

Architecture (docs/plan_a_dev_test_plan.md section 2):
  - the time-stepping loop lives in the Python host layer (skyhook/sim.py);
    every kernel here is loop-free, which keeps reverse-mode autodiff (D2+)
    free of the dynamic-loop AD restrictions;
  - the RHS follows docs/plan_a_skyhook_model.md section 3 with gamma_sky = 0:
      v1dot = -gamma_t*(v1 - zrdot) - omega_t^2*(x1 - z_r)
              - (omega_s^2/mu)*(x1 - x2) - (gamma_s/mu)*(v1 - v2)
      v2dot = -omega_s^2*(x2 - x1) - gamma_s*(v2 - v1)
  - roads are evaluated analytically inside the kernel (closed form at any t,
    RK4 half steps included) and are gradient-detached by construction.

Record channels are indexed at whole steps t_n = n*dt:
  traj[n, :]  = state (x1, x2, v1, v2) at t_n        (record-before-step)
  acc_rec[n]  = v2dot(t_n)                            (J_acc definition)
  fd_rec[n]   = suspension damper force F_d(t_n) = gamma_s*(v2 - v1)
The state after the last step is kept in the 0-d state fields.
"""

import quadrants as qd

from ._runtime import ensure_init
from .road import ROAD_SINE, ROAD_ZERO, ROAD_BUMP  # noqa: F401  (re-export)

ensure_init()

PI = 3.141592653589793

# Maximum number of recorded steps (256 s at dt = 1 ms).
T_MAX = 262144

# --- state fields (needs_grad: intermediate chain for reverse-mode AD) ------
x1 = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
x2 = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
v1 = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
v2 = qd.field(dtype=qd.f64, shape=(), needs_grad=True)

# --- per-step state history (differentiable rollout; see rollout_record) ----
# x_hist[n] = (x1, x2, v1, v2) at t_n; written once, never overwritten, so
# reverse mode always finds intact primal values (classic global-data-access
# rule). Empirically REQUIRED: adstack does NOT restore the pre-overwrite
# value of a 0-d global read-then-stored inside one kernel (T5 probe, D2).
x_hist = qd.field(dtype=qd.f64, shape=(T_MAX, 4), needs_grad=True)

# --- record channels --------------------------------------------------------
# acc_rec participates in J = RMS(acc_rec), hence needs_grad for the
# adstack reverse path (T5/T13); traj/fd_rec are diagnostic only.
traj = qd.field(dtype=qd.f64, shape=(T_MAX, 4))
acc_rec = qd.field(dtype=qd.f64, shape=(T_MAX,), needs_grad=True)
fd_rec = qd.field(dtype=qd.f64, shape=(T_MAX,))

# --- parameter fields (set per run; gradient path needs_grad) ---------------
p_mu = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
p_wtsq = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
p_wsq = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
p_gt = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
p_gs = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
p_gsky = qd.field(dtype=qd.f64, shape=(), needs_grad=True)

# --- J_acc reduction (adstack-legal pattern: accumulate inside the loop) ----
j_acc_sq = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
j_val = qd.field(dtype=qd.f64, shape=(), needs_grad=True)

# --- RHS probe fields (T6 structural assertion) -----------------------------
a1_probe = qd.field(dtype=qd.f64, shape=())
a2_probe = qd.field(dtype=qd.f64, shape=())

# --- road probe fields (kernel/python cross-check tests) -------------------
t_probe = qd.field(dtype=qd.f64, shape=1024)
z_probe = qd.field(dtype=qd.f64, shape=1024)
zd_probe = qd.field(dtype=qd.f64, shape=1024)


@qd.func
def road_eval(t: qd.f64, kind: qd.i32, amp: qd.f64, w: qd.f64,
              t0: qd.f64, width: qd.f64, ramp_t: qd.f64):
    """Analytic road (z_r, zdot_r); mirrors skyhook/road.py exactly.

    Branch-free formulation (qd.select evaluates both sides, so all divisions
    are guarded and unused branches receive safe default arguments).
    """
    # --- sine branch: z = amp*sin(w t)*e(t), smoothstep ramp e(t) ---
    has_ramp = ramp_t > 0.0
    r_safe = qd.select(has_ramp, ramp_t, 1.0)
    s_raw = t / r_safe
    s = qd.select(s_raw > 1.0, 1.0, s_raw)
    s = qd.select(s < 0.0, 0.0, s)
    s = qd.select(has_ramp, s, 1.0)
    e = 3.0 * s * s - 2.0 * s * s * s
    ramping = has_ramp & (t > 0.0) & (t < ramp_t)
    de = qd.select(ramping, (6.0 * s - 6.0 * s * s) / r_safe, 0.0)
    z_s = amp * qd.sin(w * t) * e
    zd_s = amp * (w * qd.cos(w * t) * e + qd.sin(w * t) * de)

    # --- bump branch: half-sine on [t0, t0+width] ---
    inside = (t >= t0) & (t <= t0 + width)
    ph = PI * (t - t0) / width
    z_b = qd.select(inside, amp * qd.sin(ph), 0.0)
    zd_b = qd.select(inside, amp * (PI / width) * qd.cos(ph), 0.0)

    # --- combine by kind (ROAD_ZERO -> 0) ---
    z = qd.select(kind == ROAD_SINE, z_s, qd.select(kind == ROAD_BUMP, z_b, 0.0))
    zd = qd.select(kind == ROAD_SINE, zd_s, qd.select(kind == ROAD_BUMP, zd_b, 0.0))
    return z, zd


@qd.func
def rhs_a1(x1_: qd.f64, x2_: qd.f64, v1_: qd.f64, v2_: qd.f64,
           z_r: qd.f64, zrdot: qd.f64):
    """Unsprung-mass acceleration (per unit mu)."""
    mu = p_mu[None]
    wtsq = p_wtsq[None]
    wsq = p_wsq[None]
    gt = p_gt[None]
    gs = p_gs[None]
    return (-gt * (v1_ - zrdot)
            - wtsq * (x1_ - z_r)
            - (wsq / mu) * (x1_ - x2_)
            - (gs / mu) * (v1_ - v2_))


@qd.func
def rhs_a2(x1_: qd.f64, x2_: qd.f64, v1_: qd.f64, v2_: qd.f64):
    """Sprung-mass acceleration (m2 = 1 gauge).

    Model I adds the ideal-skyhook term -gamma_sky * v2, exclusive to this
    row (the virtual damper reacts against the inertial frame, T6).
    """
    wsq = p_wsq[None]
    gs = p_gs[None]
    gsky = p_gsky[None]
    return -(wsq * (x2_ - x1_) + gs * (v2_ - v1_) + gsky * v2_)


@qd.kernel
def set_state(ix1: qd.f64, ix2: qd.f64, iv1: qd.f64, iv2: qd.f64):
    x1[None] = ix1
    x2[None] = ix2
    v1[None] = iv1
    v2[None] = iv2


@qd.kernel
def set_state_hist(ix1: qd.f64, ix2: qd.f64, iv1: qd.f64, iv2: qd.f64):
    """Initial condition for the differentiable rollout (x_hist[0])."""
    x_hist[0, 0] = ix1
    x_hist[0, 1] = ix2
    x_hist[0, 2] = iv1
    x_hist[0, 3] = iv2


@qd.kernel
def set_params(mu: qd.f64, wtsq: qd.f64, wsq: qd.f64, gt: qd.f64, gs: qd.f64,
               gsky: qd.f64):
    p_mu[None] = mu
    p_wtsq[None] = wtsq
    p_wsq[None] = wsq
    p_gt[None] = gt
    p_gs[None] = gs
    p_gsky[None] = gsky


@qd.kernel
def rk4_step(t: qd.f64, dt: qd.f64, n: qd.i32, kind: qd.i32,
             amp: qd.f64, w: qd.f64, t0: qd.f64, width: qd.f64, ramp_t: qd.f64):
    """Advance the state fields by one RK4 step; record channels at t_n."""
    x1_0 = x1[None]
    x2_0 = x2[None]
    v1_0 = v1[None]
    v2_0 = v2[None]

    # whole-step records (t_n)
    traj[n, 0] = x1_0
    traj[n, 1] = x2_0
    traj[n, 2] = v1_0
    traj[n, 3] = v2_0
    acc_rec[n] = rhs_a2(x1_0, x2_0, v1_0, v2_0)
    fd_rec[n] = p_gs[None] * (v2_0 - v1_0)

    # stage 1 at t
    z, zd = road_eval(t, kind, amp, w, t0, width, ramp_t)
    k1_x1 = v1_0
    k1_x2 = v2_0
    k1_v1 = rhs_a1(x1_0, x2_0, v1_0, v2_0, z, zd)
    k1_v2 = rhs_a2(x1_0, x2_0, v1_0, v2_0)

    # stage 2 at t + dt/2
    th = t + 0.5 * dt
    z, zd = road_eval(th, kind, amp, w, t0, width, ramp_t)
    x1s = x1_0 + 0.5 * dt * k1_x1
    x2s = x2_0 + 0.5 * dt * k1_x2
    v1s = v1_0 + 0.5 * dt * k1_v1
    v2s = v2_0 + 0.5 * dt * k1_v2
    k2_x1 = v1s
    k2_x2 = v2s
    k2_v1 = rhs_a1(x1s, x2s, v1s, v2s, z, zd)
    k2_v2 = rhs_a2(x1s, x2s, v1s, v2s)

    # stage 3 at t + dt/2
    x1s = x1_0 + 0.5 * dt * k2_x1
    x2s = x2_0 + 0.5 * dt * k2_x2
    v1s = v1_0 + 0.5 * dt * k2_v1
    v2s = v2_0 + 0.5 * dt * k2_v2
    k3_x1 = v1s
    k3_x2 = v2s
    k3_v1 = rhs_a1(x1s, x2s, v1s, v2s, z, zd)
    k3_v2 = rhs_a2(x1s, x2s, v1s, v2s)

    # stage 4 at t + dt
    te = t + dt
    z, zd = road_eval(te, kind, amp, w, t0, width, ramp_t)
    x1e = x1_0 + dt * k3_x1
    x2e = x2_0 + dt * k3_x2
    v1e = v1_0 + dt * k3_v1
    v2e = v2_0 + dt * k3_v2
    k4_x1 = v1e
    k4_x2 = v2e
    k4_v1 = rhs_a1(x1e, x2e, v1e, v2e, z, zd)
    k4_v2 = rhs_a2(x1e, x2e, v1e, v2e)

    # combine
    x1[None] = x1_0 + (dt / 6.0) * (k1_x1 + 2.0 * k2_x1 + 2.0 * k3_x1 + k4_x1)
    x2[None] = x2_0 + (dt / 6.0) * (k1_x2 + 2.0 * k2_x2 + 2.0 * k3_x2 + k4_x2)
    v1[None] = v1_0 + (dt / 6.0) * (k1_v1 + 2.0 * k2_v1 + 2.0 * k3_v1 + k4_v1)
    v2[None] = v2_0 + (dt / 6.0) * (k1_v2 + 2.0 * k2_v2 + 2.0 * k3_v2 + k4_v2)


@qd.kernel
def euler_step(t: qd.f64, dt: qd.f64, n: qd.i32, kind: qd.i32,
               amp: qd.f64, w: qd.f64, t0: qd.f64, width: qd.f64, ramp_t: qd.f64):
    """Advance the state fields by one SEMI-IMPLICIT (symplectic) Euler step.

    Convergence-order companion of rk4_step (plan D4, T10): both accelerations
    are evaluated at the current state, velocities update first, and positions
    use the NEW velocities. Same whole-step record channels as rk4_step, so
    run_passive_euler in skyhook/sim.py mirrors run_passive exactly.
    """
    x1_0 = x1[None]
    x2_0 = x2[None]
    v1_0 = v1[None]
    v2_0 = v2[None]

    # whole-step records (t_n)
    traj[n, 0] = x1_0
    traj[n, 1] = x2_0
    traj[n, 2] = v1_0
    traj[n, 3] = v2_0
    acc_rec[n] = rhs_a2(x1_0, x2_0, v1_0, v2_0)
    fd_rec[n] = p_gs[None] * (v2_0 - v1_0)

    # accelerations at the current state
    z, zd = road_eval(t, kind, amp, w, t0, width, ramp_t)
    a1 = rhs_a1(x1_0, x2_0, v1_0, v2_0, z, zd)
    a2 = rhs_a2(x1_0, x2_0, v1_0, v2_0)

    # velocities first, then positions with the new velocities
    v1[None] = v1_0 + dt * a1
    v2[None] = v2_0 + dt * a2
    x1[None] = x1_0 + dt * v1[None]
    x2[None] = x2_0 + dt * v2[None]


@qd.kernel
def rollout_record(n_steps: qd.i32, dt: qd.f64, kind: qd.i32, amp: qd.f64,
                   w: qd.f64, t0: qd.f64, width: qd.f64, ramp_t: qd.f64):
    """Whole time loop inside a single kernel (differentiable rollout path).

    Loop-carried state lives in x_hist slots, each written exactly once and
    never overwritten: every global element satisfies the classic reverse-mode
    global-data-access rule, so backward always reads intact primal values.
    (Two empirically broken alternatives, see D2 T5 probe: per-step kernel
    launches lose earlier 0-d state values across launches, and a 0-d
    read-then-store loop-carried chain gets wrong gradients even with the
    adstack enabled.) Step body identical to rk4_step; initial state must be
    written to x_hist[0] via set_state_hist beforehand. Final state lands in
    x_hist[n_steps].

    The single-iteration outer loop is required by the reverse-mode validator:
    it makes the inner dynamic range adstack-eligible so the cross-iteration
    x_hist read-after-write (read slot n, wrote slot n at the previous
    iteration) gets a proper adjoint chain instead of being rejected.
    """
    for _ in range(1):
        for n in range(n_steps):
            x1_0 = x_hist[n, 0]
            x2_0 = x_hist[n, 1]
            v1_0 = x_hist[n, 2]
            v2_0 = x_hist[n, 3]

            # whole-step records (t_n)
            traj[n, 0] = x1_0
            traj[n, 1] = x2_0
            traj[n, 2] = v1_0
            traj[n, 3] = v2_0
            acc_rec[n] = rhs_a2(x1_0, x2_0, v1_0, v2_0)
            fd_rec[n] = p_gs[None] * (v2_0 - v1_0)

            # stage 1 at t_n = n*dt
            t = n * dt
            z, zd = road_eval(t, kind, amp, w, t0, width, ramp_t)
            k1_x1 = v1_0
            k1_x2 = v2_0
            k1_v1 = rhs_a1(x1_0, x2_0, v1_0, v2_0, z, zd)
            k1_v2 = rhs_a2(x1_0, x2_0, v1_0, v2_0)

            # stage 2 at t + dt/2
            th = t + 0.5 * dt
            z, zd = road_eval(th, kind, amp, w, t0, width, ramp_t)
            x1s = x1_0 + 0.5 * dt * k1_x1
            x2s = x2_0 + 0.5 * dt * k1_x2
            v1s = v1_0 + 0.5 * dt * k1_v1
            v2s = v2_0 + 0.5 * dt * k1_v2
            k2_x1 = v1s
            k2_x2 = v2s
            k2_v1 = rhs_a1(x1s, x2s, v1s, v2s, z, zd)
            k2_v2 = rhs_a2(x1s, x2s, v1s, v2s)

            # stage 3 at t + dt/2
            x1s = x1_0 + 0.5 * dt * k2_x1
            x2s = x2_0 + 0.5 * dt * k2_x2
            v1s = v1_0 + 0.5 * dt * k2_v1
            v2s = v2_0 + 0.5 * dt * k2_v2
            k3_x1 = v1s
            k3_x2 = v2s
            k3_v1 = rhs_a1(x1s, x2s, v1s, v2s, z, zd)
            k3_v2 = rhs_a2(x1s, x2s, v1s, v2s)

            # stage 4 at t + dt
            te = t + dt
            z, zd = road_eval(te, kind, amp, w, t0, width, ramp_t)
            x1e = x1_0 + dt * k3_x1
            x2e = x2_0 + dt * k3_x2
            v1e = v1_0 + dt * k3_v1
            v2e = v2_0 + dt * k3_v2
            k4_x1 = v1e
            k4_x2 = v2e
            k4_v1 = rhs_a1(x1e, x2e, v1e, v2e, z, zd)
            k4_v2 = rhs_a2(x1e, x2e, v1e, v2e)

            # combine (each x_hist slot written once, then never touched again)
            x_hist[n + 1, 0] = x1_0 + (dt / 6.0) * (k1_x1 + 2.0 * k2_x1 + 2.0 * k3_x1 + k4_x1)
            x_hist[n + 1, 1] = x2_0 + (dt / 6.0) * (k1_x2 + 2.0 * k2_x2 + 2.0 * k3_x2 + k4_x2)
            x_hist[n + 1, 2] = v1_0 + (dt / 6.0) * (k1_v1 + 2.0 * k2_v1 + 2.0 * k3_v1 + k4_v1)
            x_hist[n + 1, 3] = v2_0 + (dt / 6.0) * (k1_v2 + 2.0 * k2_v2 + 2.0 * k3_v2 + k4_v2)


@qd.kernel
def probe_road(n: qd.i32, kind: qd.i32, amp: qd.f64, w: qd.f64,
               t0: qd.f64, width: qd.f64, ramp_t: qd.f64):
    """Evaluate road_eval at t_probe[0:n] into (z_probe, zd_probe)."""
    for i in range(n):
        z, zd = road_eval(t_probe[i], kind, amp, w, t0, width, ramp_t)
        z_probe[i] = z
        zd_probe[i] = zd


@qd.kernel
def clear_j():
    j_acc_sq[None] = 0.0


@qd.kernel
def reduce_j(n: qd.i32):
    """j_acc_sq = sum_i acc_rec[i]^2 over the first n whole-step records.

    AD-legal pattern: accumulation happens inside the loop; the clear lives
    in a separate loop-free kernel (see D0 finding on mixed loop/store).
    """
    for i in range(n):
        j_acc_sq[None] += acc_rec[i] * acc_rec[i]


@qd.kernel
def finalize_j(n: qd.i32):
    """J_acc = RMS(v2dot) over n whole-step records (contract rule 4)."""
    j_val[None] = qd.sqrt(j_acc_sq[None] / n)


@qd.kernel
def probe_rhs(t: qd.f64, kind: qd.i32, amp: qd.f64, w: qd.f64,
              t0: qd.f64, width: qd.f64, ramp_t: qd.f64):
    """Evaluate the RHS at the current state fields (T6 structural check)."""
    z, zd = road_eval(t, kind, amp, w, t0, width, ramp_t)
    a1_probe[None] = rhs_a1(x1[None], x2[None], v1[None], v2[None], z, zd)
    a2_probe[None] = rhs_a2(x1[None], x2[None], v1[None], v2[None])
