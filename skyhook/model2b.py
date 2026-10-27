"""Batched model-II kernels: (B, T) semi-active rollouts (plan A, D4).

Interface contract rule 5: states, roads and metrics are batched (B, .);
roads may differ per batch element. The three tunable gains (gamma_sky,
c_min, c_max) are per-batch fields -- the population layout the D5
optimizer samples over; the hardware-given gains (mu, omega_t^2, omega_s^2,
gamma_t) are shared scalars. Batch element b uses road fields rk_b[b]/...,
so each trajectory sees its own road.

Same reverse-mode skeleton as model2 (empirical D2/D3 finding, mandatory):
one kernel holds the whole time loop, per-step states live in xh_b slots
[b, n, :] written exactly once, and the single-iteration outer loop keeps
the inner dynamic range adstack-eligible. The batch loop b sits INNERMOST:
its iterations touch disjoint slots and carry no state, so the cross-iteration
read-after-write that matters is still the n loop's (same as the scalar case).

J path (contract rule 4, whole-step v2dot records):
    j_acc_sq_b[b] = sum_n acc_b[b, n]^2  -> j_val_b[b] = sqrt(/n)
    j_mean = sum_b j_val_b[b] / B        (scalar loss for Tape/FwdMode)
All J-path fields carry needs_grad + needs_dual (reverse and JVP routes).
"""

import quadrants as qd

from .model import road_eval  # single source: same road math as model I

B_MAX = 64
T_B_MAX = 4096  # record slots per batch element (n_steps <= T_B_MAX - 1)

# --- per-step state history (differentiable rollout) -------------------------
xh_b = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX, 4),
               needs_grad=True, needs_dual=True)

# --- record channels ----------------------------------------------------------
traj_b = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX, 4))
acc_b = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX),
                 needs_grad=True, needs_dual=True)
fd_b = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX))
mask_b = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX))
ft_b = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX))

# --- parameters: per-batch tunable trio + shared hardware gains ----------------
pg_b = qd.field(dtype=qd.f64, shape=(B_MAX,), needs_grad=True, needs_dual=True)
pcmin_b = qd.field(dtype=qd.f64, shape=(B_MAX,), needs_grad=True, needs_dual=True)
pcmax_b = qd.field(dtype=qd.f64, shape=(B_MAX,), needs_grad=True, needs_dual=True)
pmu_b = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
pwtsq_b = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
pwsq_b = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
pgt_b = qd.field(dtype=qd.f64, shape=(), needs_grad=True)

# --- per-batch road parameters -------------------------------------------------
rk_b = qd.field(dtype=qd.i32, shape=(B_MAX,))
ra_b = qd.field(dtype=qd.f64, shape=(B_MAX,))
rw_b = qd.field(dtype=qd.f64, shape=(B_MAX,))
rt0_b = qd.field(dtype=qd.f64, shape=(B_MAX,))
rwidth_b = qd.field(dtype=qd.f64, shape=(B_MAX,))
rramp_b = qd.field(dtype=qd.f64, shape=(B_MAX,))

# --- J reduction ----------------------------------------------------------------
j_acc_sq_b = qd.field(dtype=qd.f64, shape=(B_MAX,),
                      needs_grad=True, needs_dual=True)
j_val_b = qd.field(dtype=qd.f64, shape=(B_MAX,), needs_grad=True, needs_dual=True)
j_mean = qd.field(dtype=qd.f64, shape=(), needs_grad=True, needs_dual=True)


@qd.func
def fd_clip_b(b: qd.i32, v1_: qd.f64, v2_: qd.f64):
    """Force-domain clip with per-batch gains (same math as model2.fd_clip)."""
    gsky = pg_b[b]
    cmin = pcmin_b[b]
    cmax = pcmax_b[b]
    f_sky = gsky * v2_
    vr = v2_ - v1_
    lo = qd.min(cmin * vr, cmax * vr)
    hi = qd.max(cmin * vr, cmax * vr)
    return qd.min(qd.max(f_sky, lo), hi)


@qd.func
def clip_mask_b(b: qd.i32, v1_: qd.f64, v2_: qd.f64):
    """1.0 where F_sky left [lo, hi] (rule 3 diagnostics), per batch element."""
    gsky = pg_b[b]
    cmin = pcmin_b[b]
    cmax = pcmax_b[b]
    f_sky = gsky * v2_
    vr = v2_ - v1_
    lo = qd.min(cmin * vr, cmax * vr)
    hi = qd.max(cmin * vr, cmax * vr)
    return qd.select((f_sky < lo) | (f_sky > hi), 1.0, 0.0)


@qd.func
def rhs_a1_b(x1_: qd.f64, x2_: qd.f64, v1_: qd.f64, v2_: qd.f64,
             z_r: qd.f64, zrdot: qd.f64, f_d: qd.f64):
    """Unsprung-mass acceleration (per unit mu, shared gains)."""
    mu = pmu_b[None]
    wtsq = pwtsq_b[None]
    wsq = pwsq_b[None]
    gt = pgt_b[None]
    return (-gt * (v1_ - zrdot)
            - wtsq * (x1_ - z_r)
            - (wsq / mu) * (x1_ - x2_)
            + f_d / mu)


@qd.func
def rhs_a2_b(x1_: qd.f64, x2_: qd.f64, f_d: qd.f64):
    """Sprung-mass acceleration (m2 = 1 gauge, shared gains)."""
    wsq = pwsq_b[None]
    return -(wsq * (x2_ - x1_) + f_d)


@qd.kernel
def set_shared_params_b(mu: qd.f64, wtsq: qd.f64, wsq: qd.f64, gt: qd.f64):
    pmu_b[None] = mu
    pwtsq_b[None] = wtsq
    pwsq_b[None] = wsq
    pgt_b[None] = gt


@qd.kernel
def set_params_b(b: qd.i32, gsky: qd.f64, cmin: qd.f64, cmax: qd.f64):
    pg_b[b] = gsky
    pcmin_b[b] = cmin
    pcmax_b[b] = cmax


@qd.kernel
def set_road_b(b: qd.i32, kind: qd.i32, amp: qd.f64, w: qd.f64,
               t0: qd.f64, width: qd.f64, ramp_t: qd.f64):
    rk_b[b] = kind
    ra_b[b] = amp
    rw_b[b] = w
    rt0_b[b] = t0
    rwidth_b[b] = width
    rramp_b[b] = ramp_t


@qd.kernel
def set_state_hist_b(b: qd.i32, ix1: qd.f64, ix2: qd.f64,
                     iv1: qd.f64, iv2: qd.f64):
    xh_b[b, 0, 0] = ix1
    xh_b[b, 0, 1] = ix2
    xh_b[b, 0, 2] = iv1
    xh_b[b, 0, 3] = iv2


@qd.kernel
def rollout_semi_batch(n_steps: qd.i32, dt: qd.f64, B: qd.i32):
    """Batched model-II rollout: time loop outer, batch loop inner.

    Same skeleton as model2.rollout_semi per batch element; every (b, n)
    slot of xh_b is written exactly once. The innermost b loop carries no
    state (disjoint slots), so the adstack eligibility of the n loop's
    cross-iteration read-after-write is unaffected (D4 probe-verified).
    """
    for _ in range(1):
        for n in range(n_steps):
            for b in range(B):
                x1_0 = xh_b[b, n, 0]
                x2_0 = xh_b[b, n, 1]
                v1_0 = xh_b[b, n, 2]
                v2_0 = xh_b[b, n, 3]

                # whole-step records (t_n)
                traj_b[b, n, 0] = x1_0
                traj_b[b, n, 1] = x2_0
                traj_b[b, n, 2] = v1_0
                traj_b[b, n, 3] = v2_0
                acc_b[b, n] = rhs_a2_b(x1_0, x2_0, fd_clip_b(b, v1_0, v2_0))
                fd_b[b, n] = fd_clip_b(b, v1_0, v2_0)
                mask_b[b, n] = clip_mask_b(b, v1_0, v2_0)

                # stage 1 at t_n = n*dt (road at t_n shared with ft record)
                t = n * dt
                z, zd = road_eval(t, rk_b[b], ra_b[b], rw_b[b], rt0_b[b],
                                  rwidth_b[b], rramp_b[b])
                ft_b[b, n] = pmu_b[None] * pwtsq_b[None] * (x1_0 - z)
                f1 = fd_clip_b(b, v1_0, v2_0)
                k1_x1 = v1_0
                k1_x2 = v2_0
                k1_v1 = rhs_a1_b(x1_0, x2_0, v1_0, v2_0, z, zd, f1)
                k1_v2 = rhs_a2_b(x1_0, x2_0, f1)

                # stage 2 at t + dt/2
                th = t + 0.5 * dt
                z, zd = road_eval(th, rk_b[b], ra_b[b], rw_b[b], rt0_b[b],
                                  rwidth_b[b], rramp_b[b])
                x1s = x1_0 + 0.5 * dt * k1_x1
                x2s = x2_0 + 0.5 * dt * k1_x2
                v1s = v1_0 + 0.5 * dt * k1_v1
                v2s = v2_0 + 0.5 * dt * k1_v2
                f2 = fd_clip_b(b, v1s, v2s)
                k2_x1 = v1s
                k2_x2 = v2s
                k2_v1 = rhs_a1_b(x1s, x2s, v1s, v2s, z, zd, f2)
                k2_v2 = rhs_a2_b(x1s, x2s, f2)

                # stage 3 at t + dt/2
                x1s = x1_0 + 0.5 * dt * k2_x1
                x2s = x2_0 + 0.5 * dt * k2_x2
                v1s = v1_0 + 0.5 * dt * k2_v1
                v2s = v2_0 + 0.5 * dt * k2_v2
                f3 = fd_clip_b(b, v1s, v2s)
                k3_x1 = v1s
                k3_x2 = v2s
                k3_v1 = rhs_a1_b(x1s, x2s, v1s, v2s, z, zd, f3)
                k3_v2 = rhs_a2_b(x1s, x2s, f3)

                # stage 4 at t + dt
                te = t + dt
                z, zd = road_eval(te, rk_b[b], ra_b[b], rw_b[b], rt0_b[b],
                                  rwidth_b[b], rramp_b[b])
                x1e = x1_0 + dt * k3_x1
                x2e = x2_0 + dt * k3_x2
                v1e = v1_0 + dt * k3_v1
                v2e = v2_0 + dt * k3_v2
                f4 = fd_clip_b(b, v1e, v2e)
                k4_x1 = v1e
                k4_x2 = v2e
                k4_v1 = rhs_a1_b(x1e, x2e, v1e, v2e, z, zd, f4)
                k4_v2 = rhs_a2_b(x1e, x2e, f4)

                # combine (each (b, n+1) slot written once, never touched again)
                xh_b[b, n + 1, 0] = x1_0 + (dt / 6.0) * (k1_x1 + 2.0 * k2_x1 + 2.0 * k3_x1 + k4_x1)
                xh_b[b, n + 1, 1] = x2_0 + (dt / 6.0) * (k1_x2 + 2.0 * k2_x2 + 2.0 * k3_x2 + k4_x2)
                xh_b[b, n + 1, 2] = v1_0 + (dt / 6.0) * (k1_v1 + 2.0 * k2_v1 + 2.0 * k3_v1 + k4_v1)
                xh_b[b, n + 1, 3] = v2_0 + (dt / 6.0) * (k1_v2 + 2.0 * k2_v2 + 2.0 * k3_v2 + k4_v2)


@qd.kernel
def clear_j_b():
    for b in range(B_MAX):
        j_acc_sq_b[b] = 0.0


@qd.kernel
def reduce_j_b(n_steps: qd.i32, B: qd.i32):
    """j_acc_sq_b[b] = sum_n acc_b[b, n]^2 (accumulate inside the loop)."""
    for b in range(B):
        for n in range(n_steps):
            j_acc_sq_b[b] += acc_b[b, n] * acc_b[b, n]


@qd.kernel
def finalize_j_b(n_steps: qd.i32, B: qd.i32):
    """j_val_b[b] = RMS(acc_b[b, :]) over n_steps whole-step records."""
    for b in range(B):
        j_val_b[b] = qd.sqrt(j_acc_sq_b[b] / n_steps)


@qd.kernel
def clear_mean_j_b():
    j_mean[None] = 0.0


@qd.kernel
def reduce_mean_j_b(B: qd.i32):
    """j_mean = mean_b j_val_b[b] (scalar loss; accumulate inside the loop)."""
    for b in range(B):
        j_mean[None] += j_val_b[b] / B
