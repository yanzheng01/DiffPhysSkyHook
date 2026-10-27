"""Quadrants kernels for the semi-active quarter-car (plan A, D3, model II).

Architecture (docs/plan_a_dev_test_plan.md section 2,
docs/plan_a_skyhook_model.md section 4): the paired passive suspension
damper of model I is replaced by the semi-active device force, the
force-domain clip of the ideal skyhook command:
    F_sky = gamma_sky * v2                      (ideal command)
    v_rel = v2 - v1
    F_d   = clip(F_sky, min(c_min,c_max)*v_rel, max(c_min,c_max)*v_rel)
F_d enters BOTH rows pairwise (equal and opposite; no separate gamma_s):
    v1dot = -gamma_t*(v1 - zrdot) - omega_t^2*(x1 - z_r)
            - (omega_s^2/mu)*(x1 - x2) + F_d/mu
    v2dot = -omega_s^2*(x2 - x1) - F_d
The ON-OFF baseline (section 4.3) replaces the clip law by
    c(t) = c_max if v2*(v2 - v1) > 0 else c_min,   F_d = c(t) * v_rel
It is forward-only: the switching law is discontinuous, its gradient is
undefined. sim2 launches it through a qd.ad.grad_replaced wrapper whose
custom gradient is identically zero (a loud, structural zero instead of
silently differentiating through the select branch).

Rollout pattern (empirical D2/T5 finding, mandatory):
  - the whole time loop lives in ONE kernel; per-step state occupies x_hist2
    slots, each written exactly once and never overwritten (intact primal
    values for reverse mode); the single-iteration outer loop makes the
    inner dynamic range adstack-eligible for the reverse-mode validator.
  - dual-enabled (needs_dual) fields trace the forward tangent path for the
    built-in qd.ad.FwdMode JVP route (T13).

Record channels are indexed at whole steps t_n = n*dt (record-before-step):
  traj2[n, :]  = state (x1, x2, v1, v2) at t_n
  acc_rec2[n]  = v2dot(t_n) including F_d(t_n)   (J_acc channel, rule 4)
  fd_rec2[n]   = F_d(t_n)
  mask_rec2[n] = 1 where F_sky left the admissible force window (rule 3)
  ft_rec2[n]   = mu*omega_t^2*(x1 - z_r(t_n))    (tire-force channel)
  ct_rec2[n]   = c(t_n)                          (ON-OFF branch only)
"""

import quadrants as qd

from .model import T_MAX, road_eval  # single source: same road math as model I

# --- per-step state history (differentiable rollout; see module docstring) ---
x_hist2 = qd.field(dtype=qd.f64, shape=(T_MAX, 4),
                   needs_grad=True, needs_dual=True)

# --- record channels ---------------------------------------------------------
traj2 = qd.field(dtype=qd.f64, shape=(T_MAX, 4))
acc_rec2 = qd.field(dtype=qd.f64, shape=(T_MAX,), needs_grad=True, needs_dual=True)
fd_rec2 = qd.field(dtype=qd.f64, shape=(T_MAX,))
mask_rec2 = qd.field(dtype=qd.f64, shape=(T_MAX,))
ft_rec2 = qd.field(dtype=qd.f64, shape=(T_MAX,))
ct_rec2 = qd.field(dtype=qd.f64, shape=(T_MAX,))

# --- parameter fields (tunable trio carries the dual path, T13) --------------
p_mu2 = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
p_wtsq2 = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
p_wsq2 = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
p_gt2 = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
p_gsky2 = qd.field(dtype=qd.f64, shape=(), needs_grad=True, needs_dual=True)
p_cmin2 = qd.field(dtype=qd.f64, shape=(), needs_grad=True, needs_dual=True)
p_cmax2 = qd.field(dtype=qd.f64, shape=(), needs_grad=True, needs_dual=True)

# --- J_acc reduction (adstack-legal pattern: accumulate inside the loop) ------
j_acc_sq2 = qd.field(dtype=qd.f64, shape=(), needs_grad=True, needs_dual=True)
j_val2 = qd.field(dtype=qd.f64, shape=(), needs_grad=True, needs_dual=True)


@qd.func
def fd_clip(v1_: qd.f64, v2_: qd.f64):
    """Force-domain clip of the ideal skyhook command (section 4.1).

    qd.min/qd.max have no qd.clip shorthand; the built-in AD of both is the
    subgradient (truncated) semantics, which is what T9/T13 validate against
    finite differences.
    """
    gsky = p_gsky2[None]
    cmin = p_cmin2[None]
    cmax = p_cmax2[None]
    f_sky = gsky * v2_
    vr = v2_ - v1_
    lo = qd.min(cmin * vr, cmax * vr)
    hi = qd.max(cmin * vr, cmax * vr)
    return qd.min(qd.max(f_sky, lo), hi)


@qd.func
def clip_mask(v1_: qd.f64, v2_: qd.f64):
    """1.0 where F_sky left [lo, hi], else 0.0 (rule 3 diagnostics)."""
    gsky = p_gsky2[None]
    cmin = p_cmin2[None]
    cmax = p_cmax2[None]
    f_sky = gsky * v2_
    vr = v2_ - v1_
    lo = qd.min(cmin * vr, cmax * vr)
    hi = qd.max(cmin * vr, cmax * vr)
    return qd.select((f_sky < lo) | (f_sky > hi), 1.0, 0.0)


@qd.func
def fd_onoff(v1_: qd.f64, v2_: qd.f64):
    """ON-OFF law (section 4.3): returns (c(t), F_d)."""
    vr = v2_ - v1_
    c = qd.select(v2_ * vr > 0.0, p_cmax2[None], p_cmin2[None])
    return c, c * vr


@qd.func
def rhs2_a1(x1_: qd.f64, x2_: qd.f64, v1_: qd.f64, v2_: qd.f64,
            z_r: qd.f64, zrdot: qd.f64, f_d: qd.f64):
    """Unsprung-mass acceleration (per unit mu); F_d reacts pairwise (+F_d/mu)."""
    mu = p_mu2[None]
    wtsq = p_wtsq2[None]
    wsq = p_wsq2[None]
    gt = p_gt2[None]
    return (-gt * (v1_ - zrdot)
            - wtsq * (x1_ - z_r)
            - (wsq / mu) * (x1_ - x2_)
            + f_d / mu)


@qd.func
def rhs2_a2(x1_: qd.f64, x2_: qd.f64, f_d: qd.f64):
    """Sprung-mass acceleration (m2 = 1 gauge); F_d reacts pairwise (-F_d)."""
    wsq = p_wsq2[None]
    return -(wsq * (x2_ - x1_) + f_d)


@qd.kernel
def set_params2(mu: qd.f64, wtsq: qd.f64, wsq: qd.f64, gt: qd.f64,
               gsky: qd.f64, cmin: qd.f64, cmax: qd.f64):
    p_mu2[None] = mu
    p_wtsq2[None] = wtsq
    p_wsq2[None] = wsq
    p_gt2[None] = gt
    p_gsky2[None] = gsky
    p_cmin2[None] = cmin
    p_cmax2[None] = cmax


@qd.kernel
def set_state_hist2(ix1: qd.f64, ix2: qd.f64, iv1: qd.f64, iv2: qd.f64):
    """Initial condition for the differentiable rollout (x_hist2[0])."""
    x_hist2[0, 0] = ix1
    x_hist2[0, 1] = ix2
    x_hist2[0, 2] = iv1
    x_hist2[0, 3] = iv2


@qd.kernel
def rollout_semi(n_steps: qd.i32, dt: qd.f64, kind: qd.i32, amp: qd.f64,
                 w: qd.f64, t0: qd.f64, width: qd.f64, ramp_t: qd.f64):
    """Whole time loop in one kernel: semi-active clip rollout (model II).

    Same skeleton as model.rollout_record (x_hist slots + single-iteration
    outer loop); the damper force is fd_clip(stage v1, stage v2) at every RK4
    stage. mask/ft records are diagnostics (non-differentiable channels).
    """
    for _ in range(1):
        for n in range(n_steps):
            x1_0 = x_hist2[n, 0]
            x2_0 = x_hist2[n, 1]
            v1_0 = x_hist2[n, 2]
            v2_0 = x_hist2[n, 3]

            # whole-step records (t_n)
            traj2[n, 0] = x1_0
            traj2[n, 1] = x2_0
            traj2[n, 2] = v1_0
            traj2[n, 3] = v2_0
            acc_rec2[n] = rhs2_a2(x1_0, x2_0, fd_clip(v1_0, v2_0))
            fd_rec2[n] = fd_clip(v1_0, v2_0)
            mask_rec2[n] = clip_mask(v1_0, v2_0)

            # stage 1 at t_n = n*dt (road at t_n shared with the ft record)
            t = n * dt
            z, zd = road_eval(t, kind, amp, w, t0, width, ramp_t)
            ft_rec2[n] = p_mu2[None] * p_wtsq2[None] * (x1_0 - z)
            f1 = fd_clip(v1_0, v2_0)
            k1_x1 = v1_0
            k1_x2 = v2_0
            k1_v1 = rhs2_a1(x1_0, x2_0, v1_0, v2_0, z, zd, f1)
            k1_v2 = rhs2_a2(x1_0, x2_0, f1)

            # stage 2 at t + dt/2
            th = t + 0.5 * dt
            z, zd = road_eval(th, kind, amp, w, t0, width, ramp_t)
            x1s = x1_0 + 0.5 * dt * k1_x1
            x2s = x2_0 + 0.5 * dt * k1_x2
            v1s = v1_0 + 0.5 * dt * k1_v1
            v2s = v2_0 + 0.5 * dt * k1_v2
            f2 = fd_clip(v1s, v2s)
            k2_x1 = v1s
            k2_x2 = v2s
            k2_v1 = rhs2_a1(x1s, x2s, v1s, v2s, z, zd, f2)
            k2_v2 = rhs2_a2(x1s, x2s, f2)

            # stage 3 at t + dt/2
            x1s = x1_0 + 0.5 * dt * k2_x1
            x2s = x2_0 + 0.5 * dt * k2_x2
            v1s = v1_0 + 0.5 * dt * k2_v1
            v2s = v2_0 + 0.5 * dt * k2_v2
            f3 = fd_clip(v1s, v2s)
            k3_x1 = v1s
            k3_x2 = v2s
            k3_v1 = rhs2_a1(x1s, x2s, v1s, v2s, z, zd, f3)
            k3_v2 = rhs2_a2(x1s, x2s, f3)

            # stage 4 at t + dt
            te = t + dt
            z, zd = road_eval(te, kind, amp, w, t0, width, ramp_t)
            x1e = x1_0 + dt * k3_x1
            x2e = x2_0 + dt * k3_x2
            v1e = v1_0 + dt * k3_v1
            v2e = v2_0 + dt * k3_v2
            f4 = fd_clip(v1e, v2e)
            k4_x1 = v1e
            k4_x2 = v2e
            k4_v1 = rhs2_a1(x1e, x2e, v1e, v2e, z, zd, f4)
            k4_v2 = rhs2_a2(x1e, x2e, f4)

            # combine (each x_hist2 slot written once, then never touched again)
            x_hist2[n + 1, 0] = x1_0 + (dt / 6.0) * (k1_x1 + 2.0 * k2_x1 + 2.0 * k3_x1 + k4_x1)
            x_hist2[n + 1, 1] = x2_0 + (dt / 6.0) * (k1_x2 + 2.0 * k2_x2 + 2.0 * k3_x2 + k4_x2)
            x_hist2[n + 1, 2] = v1_0 + (dt / 6.0) * (k1_v1 + 2.0 * k2_v1 + 2.0 * k3_v1 + k4_v1)
            x_hist2[n + 1, 3] = v2_0 + (dt / 6.0) * (k1_v2 + 2.0 * k2_v2 + 2.0 * k3_v2 + k4_v2)


@qd.kernel
def rollout_onoff(n_steps: qd.i32, dt: qd.f64, kind: qd.i32, amp: qd.f64,
                  w: qd.f64, t0: qd.f64, width: qd.f64, ramp_t: qd.f64):
    """Whole time loop in one kernel: ON-OFF baseline rollout (section 4.3).

    Forward-only: launch through the grad_replaced wrapper in sim2 (zero
    gradient); never launch this kernel inside a Tape/FwdMode directly.
    Note: do NOT unpack tuple returns into `_` here -- it collides with the
    outer `for _ in range(1)` loop variable and trips a LocalStoreStmt
    assertion in the compiler (bisected in D3).
    """
    for _ in range(1):
        for n in range(n_steps):
            x1_0 = x_hist2[n, 0]
            x2_0 = x_hist2[n, 1]
            v1_0 = x_hist2[n, 2]
            v2_0 = x_hist2[n, 3]

            # whole-step records (t_n)
            traj2[n, 0] = x1_0
            traj2[n, 1] = x2_0
            traj2[n, 2] = v1_0
            traj2[n, 3] = v2_0
            c_now, f_now = fd_onoff(v1_0, v2_0)
            acc_rec2[n] = rhs2_a2(x1_0, x2_0, f_now)
            fd_rec2[n] = f_now
            ct_rec2[n] = c_now

            # stage 1 at t_n = n*dt (road at t_n shared with the ft record)
            t = n * dt
            z, zd = road_eval(t, kind, amp, w, t0, width, ramp_t)
            ft_rec2[n] = p_mu2[None] * p_wtsq2[None] * (x1_0 - z)
            k1_x1 = v1_0
            k1_x2 = v2_0
            k1_v1 = rhs2_a1(x1_0, x2_0, v1_0, v2_0, z, zd, f_now)
            k1_v2 = rhs2_a2(x1_0, x2_0, f_now)

            # stage 2 at t + dt/2
            th = t + 0.5 * dt
            z, zd = road_eval(th, kind, amp, w, t0, width, ramp_t)
            x1s = x1_0 + 0.5 * dt * k1_x1
            x2s = x2_0 + 0.5 * dt * k1_x2
            v1s = v1_0 + 0.5 * dt * k1_v1
            v2s = v2_0 + 0.5 * dt * k1_v2
            c2, f2 = fd_onoff(v1s, v2s)
            k2_x1 = v1s
            k2_x2 = v2s
            k2_v1 = rhs2_a1(x1s, x2s, v1s, v2s, z, zd, f2)
            k2_v2 = rhs2_a2(x1s, x2s, f2)

            # stage 3 at t + dt/2
            x1s = x1_0 + 0.5 * dt * k2_x1
            x2s = x2_0 + 0.5 * dt * k2_x2
            v1s = v1_0 + 0.5 * dt * k2_v1
            v2s = v2_0 + 0.5 * dt * k2_v2
            c3, f3 = fd_onoff(v1s, v2s)
            k3_x1 = v1s
            k3_x2 = v2s
            k3_v1 = rhs2_a1(x1s, x2s, v1s, v2s, z, zd, f3)
            k3_v2 = rhs2_a2(x1s, x2s, f3)

            # stage 4 at t + dt
            te = t + dt
            z, zd = road_eval(te, kind, amp, w, t0, width, ramp_t)
            x1e = x1_0 + dt * k3_x1
            x2e = x2_0 + dt * k3_x2
            v1e = v1_0 + dt * k3_v1
            v2e = v2_0 + dt * k3_v2
            c4, f4 = fd_onoff(v1e, v2e)
            k4_x1 = v1e
            k4_x2 = v2e
            k4_v1 = rhs2_a1(x1e, x2e, v1e, v2e, z, zd, f4)
            k4_v2 = rhs2_a2(x1e, x2e, f4)

            # combine (each x_hist2 slot written once, then never touched again)
            x_hist2[n + 1, 0] = x1_0 + (dt / 6.0) * (k1_x1 + 2.0 * k2_x1 + 2.0 * k3_x1 + k4_x1)
            x_hist2[n + 1, 1] = x2_0 + (dt / 6.0) * (k1_x2 + 2.0 * k2_x2 + 2.0 * k3_x2 + k4_x2)
            x_hist2[n + 1, 2] = v1_0 + (dt / 6.0) * (k1_v1 + 2.0 * k2_v1 + 2.0 * k3_v1 + k4_v1)
            x_hist2[n + 1, 3] = v2_0 + (dt / 6.0) * (k1_v2 + 2.0 * k2_v2 + 2.0 * k3_v2 + k4_v2)


# --- clip force probe (T9 unit-level domain-split gradient check) ------------
fd_probe = qd.field(dtype=qd.f64, shape=(), needs_grad=True, needs_dual=True)


@qd.kernel
def probe_fd(v1_: qd.f64, v2_: qd.f64):
    """Evaluate F_d at an arbitrary (v1, v2) sample (T9 gradient probe)."""
    fd_probe[None] = fd_clip(v1_, v2_)


@qd.kernel
def clear_j2():
    j_acc_sq2[None] = 0.0


@qd.kernel
def reduce_j2(n: qd.i32):
    """j_acc_sq2 = sum_i acc_rec2[i]^2 over the first n whole-step records."""
    for i in range(n):
        j_acc_sq2[None] += acc_rec2[i] * acc_rec2[i]


@qd.kernel
def finalize_j2(n: qd.i32):
    """J_acc = RMS(v2dot) over n whole-step records (contract rule 4)."""
    j_val2[None] = qd.sqrt(j_acc_sq2[None] / n)
