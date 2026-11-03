"""Model-III kernels: discrete-road command-driven rollout + composite MS
loss (plan C, F0 -- docs/plan_c_policy_training_dev_test_plan.md section 3).

Increment discipline (plan C section 2-5): model/model2/model2b are FROZEN;
every field/kernel here is new. The rollout skeleton is copied verbatim from
model2b.rollout_semi_cmd_batch (the D2/T5-validated reverse-mode pattern:
one kernel holds the whole time loop, every (b, n) slot of xh3 is written
exactly once, the single-iteration outer loop keeps the inner dynamic range
adstack-eligible, the batch loop sits innermost carrying no state).

What changes vs model2b:

  - Road evaluation: the analytic road_eval is replaced by road_eval_seq,
    a LINEAR INTERPOLATION over the host-written half-step grid rs_z3 /
    rs_zd3 (plan A contract rule 1: discrete spectra interpolate; the
    kernel NEVER differences the discrete series). The grid coordinate
    s = t/(dt/2) is built from INTEGER index arithmetic (i0 + 2n, +1), so
    every rollout evaluation lands exactly on a grid point (frac = 0,
    lookup exact); the interpolation exists for off-grid generality.
    i0 is the global half-step index of the segment start (receding-
    horizon segments, plan C decision 1). The sequence fields carry NO
    gradient (environment inputs, detached by design).

  - travel / ft record channels are differentiable (needs_grad +
    needs_dual): they feed the composite loss. fd3 / mask3 stay
    diagnostic (same split as model2b's acc_b vs fd_b/mask_b).

  - j_sim3 composite MS reduction (plan C decisions 3 and 8):
        J_sim[b] = ( w_acc*MS(a2) + w_disp*MS(x2-x1) + w_tire*MS(F_t)
                     + w_lim*MS(ReLU(|x2-x1| - x_max)) )[b]
    per batch element (mean over the n_steps whole-step records), then
    j_mean3 = mean_b J_sim[b] -- the scalar loss for Tape/FwdMode. x_max
    is a per-batch (B,) envelope field with NO gradient (hardware
    envelope, plan C decision 8; default 0.08 m). The hinge term is
    evaluated as the exact algebraic decomposition
        max(d - xmax, 0)^2 + max(-d - xmax, 0)^2   ==   ReLU(|d|-xmax)^2
    built from qd.max alone (T9-verified ReLU semantics; no reliance on
    the AD of abs). MS not RMS: no sqrt anywhere in the J chain, so the
    sqrt'(0) = nan dual poison of the D4 finding cannot occur by
    construction (nonzero-IC discipline kept regardless).
"""

import quadrants as qd

from ._runtime import ensure_init

ensure_init()

B_MAX = 64
T_B_MAX = 4096   # record slots per batch element (n_steps <= T_B_MAX - 1)
RS_MAX = 2 * T_B_MAX  # half-step grid points (covers n_steps <= T_B_MAX-1)

# --- per-step state history (differentiable rollout) -------------------------
xh3 = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX, 4),
               needs_grad=True, needs_dual=True)

# --- external force-command channel (same contract as model2b.fc_b) -----------
fc3 = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX),
               needs_grad=True, needs_dual=True)

# --- record channels ----------------------------------------------------------
traj3 = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX, 4))
acc3 = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX),
                needs_grad=True, needs_dual=True)     # J path (MS a2)
travel3 = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX),
                   needs_grad=True, needs_dual=True)  # J path (x2 - x1)
ft3 = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX),
               needs_grad=True, needs_dual=True)      # J path (tire force)
fd3 = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX))      # diagnostic
mask3 = qd.field(dtype=qd.f64, shape=(B_MAX, T_B_MAX))    # cmd out-of-band

# --- parameters: per-batch tunable trio + shared hardware gains ----------------
pg3 = qd.field(dtype=qd.f64, shape=(B_MAX,), needs_grad=True, needs_dual=True)
pcmin3 = qd.field(dtype=qd.f64, shape=(B_MAX,), needs_grad=True, needs_dual=True)
pcmax3 = qd.field(dtype=qd.f64, shape=(B_MAX,), needs_grad=True, needs_dual=True)
pmu3 = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
pwtsq3 = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
pwsq3 = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
pgt3 = qd.field(dtype=qd.f64, shape=(), needs_grad=True)

# --- stroke envelope: per-batch hardware field, NO gradient (decision 8) -------
xmax3 = qd.field(dtype=qd.f64, shape=(B_MAX,))

# --- discrete road sequence: half-step grid, NO gradient (detached) -----------
rs_z3 = qd.field(dtype=qd.f64, shape=(B_MAX, RS_MAX))
rs_zd3 = qd.field(dtype=qd.f64, shape=(B_MAX, RS_MAX))

# --- J_sim reduction accumulators (adstack-legal: clear kernel + in-loop adds) -
js_acc3 = qd.field(dtype=qd.f64, shape=(B_MAX,),
                   needs_grad=True, needs_dual=True)
js_disp3 = qd.field(dtype=qd.f64, shape=(B_MAX,),
                    needs_grad=True, needs_dual=True)
js_tire3 = qd.field(dtype=qd.f64, shape=(B_MAX,),
                    needs_grad=True, needs_dual=True)
js_lim3 = qd.field(dtype=qd.f64, shape=(B_MAX,),
                   needs_grad=True, needs_dual=True)
j_val3 = qd.field(dtype=qd.f64, shape=(B_MAX,), needs_grad=True, needs_dual=True)
j_mean3 = qd.field(dtype=qd.f64, shape=(), needs_grad=True, needs_dual=True)

# --- road channel probes (T19 cross-checks) ------------------------------------
pr_s = qd.field(dtype=qd.f64, shape=1024)
pr_z = qd.field(dtype=qd.f64, shape=1024)
pr_zd = qd.field(dtype=qd.f64, shape=1024)


@qd.func
def road_eval_seq(b: qd.i32, i: qd.i32, frac: qd.f64):
    """Linear interpolation of the half-step sequence at grid index i.

    Evaluates at grid coordinate s = i + frac (s = t/(dt/2), dimensionless):
    the rollout passes the INTEGER stage index with frac = 0 (grid points
    hit exactly -- qd.floor is not supported by the dual transformation,
    so the rollout path must not compute a fractional index); the probe
    kernel (non-dual) computes (i, frac) from an arbitrary s with floor.
    The upper neighbor is clamped into range so an evaluation on the very
    last grid point stays in bounds (frac is exactly 0 there, the clamp
    cannot change any rollout value).
    """
    i1 = qd.min(i + 1, RS_MAX - 1)
    z_lo = rs_z3[b, i]
    z_hi = rs_z3[b, i1]
    zd_lo = rs_zd3[b, i]
    zd_hi = rs_zd3[b, i1]
    z = z_lo + frac * (z_hi - z_lo)
    zd = zd_lo + frac * (zd_hi - zd_lo)
    return z, zd


@qd.func
def fd_clip_cmd3(b: qd.i32, v1_: qd.f64, v2_: qd.f64, fc: qd.f64):
    """Force-domain clip of the external command (same math as model2b)."""
    cmin = pcmin3[b]
    cmax = pcmax3[b]
    vr = v2_ - v1_
    lo = qd.min(cmin * vr, cmax * vr)
    hi = qd.max(cmin * vr, cmax * vr)
    return qd.min(qd.max(fc, lo), hi)


@qd.func
def clip_mask_cmd3(b: qd.i32, v1_: qd.f64, v2_: qd.f64, fc: qd.f64):
    """1.0 where the COMMAND left [lo, hi] (diagnostics)."""
    cmin = pcmin3[b]
    cmax = pcmax3[b]
    vr = v2_ - v1_
    lo = qd.min(cmin * vr, cmax * vr)
    hi = qd.max(cmin * vr, cmax * vr)
    return qd.select((fc < lo) | (fc > hi), 1.0, 0.0)


@qd.func
def rhs_a1_3(x1_: qd.f64, x2_: qd.f64, v1_: qd.f64, v2_: qd.f64,
             z_r: qd.f64, zrdot: qd.f64, f_d: qd.f64):
    """Unsprung-mass acceleration (per unit mu, shared gains)."""
    mu = pmu3[None]
    wtsq = pwtsq3[None]
    wsq = pwsq3[None]
    gt = pgt3[None]
    return (-gt * (v1_ - zrdot)
            - wtsq * (x1_ - z_r)
            - (wsq / mu) * (x1_ - x2_)
            + f_d / mu)


@qd.func
def rhs_a2_3(x1_: qd.f64, x2_: qd.f64, f_d: qd.f64):
    """Sprung-mass acceleration (m2 = 1 gauge, shared gains)."""
    wsq = pwsq3[None]
    return -(wsq * (x2_ - x1_) + f_d)


@qd.kernel
def set_shared_params3(mu: qd.f64, wtsq: qd.f64, wsq: qd.f64, gt: qd.f64):
    pmu3[None] = mu
    pwtsq3[None] = wtsq
    pwsq3[None] = wsq
    pgt3[None] = gt


@qd.kernel
def set_params3(b: qd.i32, gsky: qd.f64, cmin: qd.f64, cmax: qd.f64):
    pg3[b] = gsky
    pcmin3[b] = cmin
    pcmax3[b] = cmax


@qd.kernel
def set_xmax3(b: qd.i32, xmax: qd.f64):
    xmax3[b] = xmax


@qd.kernel
def set_state_hist3(b: qd.i32, ix1: qd.f64, ix2: qd.f64,
                    iv1: qd.f64, iv2: qd.f64):
    xh3[b, 0, 0] = ix1
    xh3[b, 0, 1] = ix2
    xh3[b, 0, 2] = iv1
    xh3[b, 0, 3] = iv2


@qd.kernel
def set_fcmd3_elem(b: qd.i32, n: qd.i32, val: qd.f64):
    """Write one command sample fc3[b, n] (element-wise host write, E4 pattern).

    Executed BEFORE the rollout launch (outside any Tape/FwdMode), so the
    primal command values are intact when the reverse pass reads them.
    """
    fc3[b, n] = val


@qd.kernel
def set_road_seq_elem(b: qd.i32, i: qd.i32, z: qd.f64, zd: qd.f64):
    """Write one half-step grid sample rs_z3[b, i] / rs_zd3[b, i].

    Element-wise host writes (model2b set_*_elem pattern); sequences are
    written ONCE per episode and read by every segment rollout via the
    i0 offset. NO gradient: the road is an environment input (detach).
    """
    rs_z3[b, i] = z
    rs_zd3[b, i] = zd


@qd.kernel
def rollout_cmd_seq_batch(n_steps: qd.i32, dt: qd.f64, B: qd.i32, i0: qd.i32):
    """Batched command-driven rollout on the discrete road sequence.

    Same skeleton as model2b.rollout_semi_cmd_batch (time loop outer,
    batch loop inner, every (b, n) xh3 slot written exactly once, ZOH
    command read once per (b, n) and held across the 4 RK4 stages). The
    road at stage times is road_eval_seq at the GLOBAL half-step index
    i0 + 2n (whole step t_n) / i0 + 2n + 1 (half step): integer
    arithmetic, so grid points are hit exactly (interpolation frac = 0).

    Records at t_n: traj3, acc3, fd3, mask3 (command out-of-band),
    travel3 = x2 - x1, ft3 = mu*omega_t^2*(x1 - z_r(t_n)).
    """
    for _ in range(1):
        for n in range(n_steps):
            for b in range(B):
                x1_0 = xh3[b, n, 0]
                x2_0 = xh3[b, n, 1]
                v1_0 = xh3[b, n, 2]
                v2_0 = xh3[b, n, 3]
                fc = fc3[b, n]   # ZOH: constant across the 4 stages of step n

                # whole-step records (t_n)
                traj3[b, n, 0] = x1_0
                traj3[b, n, 1] = x2_0
                traj3[b, n, 2] = v1_0
                traj3[b, n, 3] = v2_0
                acc3[b, n] = rhs_a2_3(x1_0, x2_0,
                                       fd_clip_cmd3(b, v1_0, v2_0, fc))
                fd3[b, n] = fd_clip_cmd3(b, v1_0, v2_0, fc)
                mask3[b, n] = clip_mask_cmd3(b, v1_0, v2_0, fc)
                travel3[b, n] = x2_0 - x1_0

                # stage 1 at t_n (global half-step index i0 + 2n)
                gi = i0 + 2 * n
                z, zd = road_eval_seq(b, gi, 0.0)
                ft3[b, n] = pmu3[None] * pwtsq3[None] * (x1_0 - z)
                f1 = fd_clip_cmd3(b, v1_0, v2_0, fc)
                k1_x1 = v1_0
                k1_x2 = v2_0
                k1_v1 = rhs_a1_3(x1_0, x2_0, v1_0, v2_0, z, zd, f1)
                k1_v2 = rhs_a2_3(x1_0, x2_0, f1)

                # stage 2 at t + dt/2 (index i0 + 2n + 1)
                z, zd = road_eval_seq(b, gi + 1, 0.0)
                x1s = x1_0 + 0.5 * dt * k1_x1
                x2s = x2_0 + 0.5 * dt * k1_x2
                v1s = v1_0 + 0.5 * dt * k1_v1
                v2s = v2_0 + 0.5 * dt * k1_v2
                f2 = fd_clip_cmd3(b, v1s, v2s, fc)
                k2_x1 = v1s
                k2_x2 = v2s
                k2_v1 = rhs_a1_3(x1s, x2s, v1s, v2s, z, zd, f2)
                k2_v2 = rhs_a2_3(x1s, x2s, f2)

                # stage 3 at t + dt/2
                x1s = x1_0 + 0.5 * dt * k2_x1
                x2s = x2_0 + 0.5 * dt * k2_x2
                v1s = v1_0 + 0.5 * dt * k2_v1
                v2s = v2_0 + 0.5 * dt * k2_v2
                f3 = fd_clip_cmd3(b, v1s, v2s, fc)
                k3_x1 = v1s
                k3_x2 = v2s
                k3_v1 = rhs_a1_3(x1s, x2s, v1s, v2s, z, zd, f3)
                k3_v2 = rhs_a2_3(x1s, x2s, f3)

                # stage 4 at t + dt (index i0 + 2n + 2)
                z, zd = road_eval_seq(b, gi + 2, 0.0)
                x1e = x1_0 + dt * k3_x1
                x2e = x2_0 + dt * k3_x2
                v1e = v1_0 + dt * k3_v1
                v2e = v2_0 + dt * k3_v2
                f4 = fd_clip_cmd3(b, v1e, v2e, fc)
                k4_x1 = v1e
                k4_x2 = v2e
                k4_v1 = rhs_a1_3(x1e, x2e, v1e, v2e, z, zd, f4)
                k4_v2 = rhs_a2_3(x1e, x2e, f4)

                # combine (each (b, n+1) slot written once, never touched again)
                xh3[b, n + 1, 0] = x1_0 + (dt / 6.0) * (k1_x1 + 2.0 * k2_x1 + 2.0 * k3_x1 + k4_x1)
                xh3[b, n + 1, 1] = x2_0 + (dt / 6.0) * (k1_x2 + 2.0 * k2_x2 + 2.0 * k3_x2 + k4_x2)
                xh3[b, n + 1, 2] = v1_0 + (dt / 6.0) * (k1_v1 + 2.0 * k2_v1 + 2.0 * k3_v1 + k4_v1)
                xh3[b, n + 1, 3] = v2_0 + (dt / 6.0) * (k1_v2 + 2.0 * k2_v2 + 2.0 * k3_v2 + k4_v2)


@qd.kernel
def probe_road_seq(b: qd.i32, n: qd.i32):
    """Evaluate road_eval_seq at pr_s[0:n] into (pr_z, pr_zd) (T19 checks).

    Non-dual kernel (no needs_grad/needs_dual fields touched), so qd.floor
    is allowed HERE to split an arbitrary grid coordinate s into the
    (integer index, fraction) pair the interpolation func expects.
    """
    for k in range(n):
        s = pr_s[k]
        i_f = qd.floor(s)
        i = qd.cast(i_f, qd.i32)
        frac = s - i_f
        z, zd = road_eval_seq(b, i, frac)
        pr_z[k] = z
        pr_zd[k] = zd


@qd.kernel
def clear_jsim3():
    for b in range(B_MAX):
        js_acc3[b] = 0.0
        js_disp3[b] = 0.0
        js_tire3[b] = 0.0
        js_lim3[b] = 0.0


@qd.kernel
def reduce_jsim3(n_steps: qd.i32, B: qd.i32):
    """Accumulate the four sum-of-squares terms (accumulate inside the loop).

    The stroke hinge uses the exact two-ReLU decomposition
        max(d - xmax, 0)^2 + max(-d - xmax, 0)^2  ==  ReLU(|d| - xmax)^2
    (qd.max semantics verified by T9; avoids depending on the AD of abs).
    """
    for b in range(B):
        for n in range(n_steps):
            a = acc3[b, n]
            d = travel3[b, n]
            ft = ft3[b, n]
            xm = xmax3[b]
            ov_p = qd.max(d - xm, 0.0)
            ov_m = qd.max(-d - xm, 0.0)
            hinge = ov_p * ov_p + ov_m * ov_m
            js_acc3[b] += a * a
            js_disp3[b] += d * d
            js_tire3[b] += ft * ft
            js_lim3[b] += hinge


@qd.kernel
def finalize_jsim3(n_steps: qd.i32, B: qd.i32,
                   w_acc: qd.f64, w_disp: qd.f64, w_tire: qd.f64, w_lim: qd.f64):
    """J_sim[b] = weighted MS combination (mean over n_steps records)."""
    for b in range(B):
        j_val3[b] = (w_acc * js_acc3[b] + w_disp * js_disp3[b]
                     + w_tire * js_tire3[b] + w_lim * js_lim3[b]) / n_steps


@qd.kernel
def clear_mean_jsim3():
    j_mean3[None] = 0.0


@qd.kernel
def reduce_mean_jsim3(B: qd.i32):
    """j_mean3 = mean_b J_sim[b] (scalar loss for Tape/FwdMode)."""
    for b in range(B):
        j_mean3[None] += j_val3[b] / B
