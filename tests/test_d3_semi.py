"""D3 model-II (semi-active clip + ON-OFF) tests (Gate: T7, T8, T9, T13).

Stage D3 of docs/plan_a_dev_test_plan.md: the passive suspension damper is
replaced by the force-domain clip of the ideal skyhook command
(docs/plan_a_skyhook_model.md section 4), plus the forward-only ON-OFF
baseline (section 4.3) with an explicit zero-gradient marking.

Anchors:
  T7  passivity audit: F_d * v_rel >= 0 at every whole step for both the clip
      and the ON-OFF law (including steps on the clip boundary), and the
      free-decay energy bookkeeping: mechanical energy non-increasing, total
      dissipated integral sum(F_d * v_rel) * dt >= 0.
  T8  degenerate limits: (a) passive degeneracy gamma_sky = 0 and
      c_min = c_max = gamma_s reproduces the model-I passive run (rel RMS
      difference < 1e-8); (b) ideal realizability with c_min = 0 and a huge
      c_max: on every step where the mask is inactive the device realises
      F_d == F_sky bit-exactly, while the remaining steps record the
      structural semi-active limitation (the clip cannot be fully opened).
  T9  domain-split gradient semantics of the clip force, probed at samples
      that are strictly in/out of the admissible band (unit level):
      in-band dF_d/dgamma_sky == v2 (the model-I ideal-skyhook force
      gradient), clipped: the gamma_sky gradient is truncated to 0 and the
      active bound takes over with dF_d/d_bound == v_rel. All three routes
      (reverse Tape, FwdMode JVP, central FD) must agree.
  T13 full-loop three-way gradient validation through the 1000-step rollout
      (mixed domain, clip active on part of the trajectory): forward JVP vs
      adstack reverse vs central FD, per tunable parameter (gamma_sky, c_min,
      c_max). NOTE on the domain split: a strictly mask-free nontrivial
      trajectory does not exist -- v2/v_rel sweeps the whole real line on
      any damped mode (phase unlock), so the clip necessarily engages. The
      honest smoothness anchor is ACTIVE-SET STABILITY: the mask arrays of
      the gamma +/- h runs are asserted identical, which makes J smooth on
      the central-FD window; the truncated (subgradient) AD then matches FD
      on both the in-band and the clipped steps (empirical D3 finding).

ON-OFF zero-gradient marking: the ON-OFF rollout is launched through a
qd.ad.grad_replaced wrapper with an identically-zero custom gradient, so a
Tape around it yields exact 0.0 parameter gradients instead of silently
differentiating through the discontinuous qd.select branch.
"""

import math

import numpy as np
import pytest
import quadrants as qd

from analytic import mechanical_energy
from skyhook import model2
from skyhook.params import PassiveParams, SemiActiveParams
from skyhook.road import BumpRoad, ZeroRoad
from skyhook.sim import run_passive
from skyhook.sim2 import _onoff_rollout, run_onoff, run_semi

P = SemiActiveParams()
DT = 1.0e-3
N_STEPS = 1000
ROAD = BumpRoad(amp=0.05, t0=0.05, width=0.1)
ROAD_ARGS = ROAD.kernel_args()


# ---------------------------------------------------------------------------
# T7: passivity audit
# ---------------------------------------------------------------------------
class TestT7Passivity:
    @pytest.mark.parametrize("runner", [run_semi, run_onoff],
                             ids=["clip", "onoff"])
    def test_fd_dot_vrel_nonneg(self, runner):
        res = runner(P, ROAD, t_end=1.0)
        v_rel = res.state[:, 3] - res.state[:, 2]
        products = res.fd * v_rel
        print(f"[D3] T7 {runner.__name__}: min F_d*v_rel = {np.min(products):.3e} "
              f"over {len(products)} steps (bump, mixed domain)")
        assert np.min(products) >= -1e-12

    def test_free_decay_energy_audit(self):
        ic = (1.0e-3, 1.2e-3, 0.0, 0.0)
        res = run_semi(P, ZeroRoad(), x0=ic, t_end=4.0)
        v_rel = res.state[:, 3] - res.state[:, 2]
        dissipated = float(np.sum(res.fd * v_rel) * DT)
        e = mechanical_energy(res.state, P)
        d_e_max = float(np.max(np.diff(e)))
        print(f"[D3] T7 free decay: sum F_d*v_rel*dt = {dissipated:.6e} (>=0), "
              f"E0 = {e[0]:.6e}, E_end = {e[-1]:.3e}, max step dE = {d_e_max:.3e}")
        assert dissipated >= -1e-12
        assert e[-1] < e[0]
        assert d_e_max <= 1e-12 * max(abs(e[0]), 1e-30)


# ---------------------------------------------------------------------------
# T8: degenerate limits
# ---------------------------------------------------------------------------
class TestT8DegenerateLimits:
    def test_passive_degeneracy(self):
        """gamma_sky = 0, c_min = c_max = gamma_s == passive model I."""
        gs = 2.0 * 0.10 * P.omega_s
        res2 = run_semi(SemiActiveParams(gamma_sky=0.0, c_min=gs, c_max=gs),
                       ROAD, t_end=1.0)
        res1 = run_passive(PassiveParams(gamma_s=gs, gamma_sky=0.0), ROAD,
                           t_end=1.0)
        rms = math.sqrt(np.mean((res2.a2 - res1.a2) ** 2))
        rel = rms / math.sqrt(np.mean(res1.a2 ** 2))
        print(f"[D3] T8a passive degeneracy: RMS(a2 diff) = {rms:.3e}, "
              f"rel = {rel:.3e}")
        assert rel < 1e-8

    def test_ideal_realizability(self):
        """c_min = 0, c_max huge: inactive steps realise F_d == F_sky exactly."""
        p = SemiActiveParams(c_min=0.0, c_max=1.0e6 * P.gamma_sky)
        res = run_semi(p, ROAD, t_end=1.0)
        f_sky = p.gamma_sky * res.state[:, 3]
        inactive = res.mask == 0
        active = int((~inactive).sum())
        err = float(np.max(np.abs(res.fd[inactive] - f_sky[inactive])))
        print(f"[D3] T8b ideal realizability: inactive steps = "
              f"{int(inactive.sum())}/{len(inactive)}, max|F_d - F_sky| on them "
              f"= {err:.3e}; active steps (structural limit) = {active}")
        assert inactive.sum() > 0          # non-vacuous claim
        assert err < 1e-14                 # bit-exact realisation
        assert active > 0                   # clip cannot be fully opened


# ---------------------------------------------------------------------------
# T9: domain-split gradient semantics of the clip force (unit level)
# ---------------------------------------------------------------------------
def clip_unit_grads(gsky, cmin, cmax, v1, v2):
    """(dF/dgamma_sky, dF/dc_min, dF/dc_max) via reverse, JVP and central FD."""
    model2.set_params2(P.mu, P.omega_t_sq, P.omega_s_sq, P.gamma_t,
                       gsky, cmin, cmax)
    with qd.ad.Tape(loss=model2.fd_probe):
        model2.probe_fd(v1, v2)
    g_ad = (model2.p_gsky2.grad[None], model2.p_cmin2.grad[None],
            model2.p_cmax2.grad[None])

    g_jvp = []
    for pf in (model2.p_gsky2, model2.p_cmin2, model2.p_cmax2):
        with qd.ad.FwdMode(loss=model2.fd_probe, param=pf):
            model2.probe_fd(v1, v2)
        g_jvp.append(model2.fd_probe.dual[None])
    g_jvp = tuple(g_jvp)

    h = 1.0e-6

    def f_of(g, c_lo, c_hi):
        model2.set_params2(P.mu, P.omega_t_sq, P.omega_s_sq, P.gamma_t,
                           g, c_lo, c_hi)
        model2.probe_fd(v1, v2)
        return model2.fd_probe[None]

    g_fd = []
    for i in range(3):
        up, dn = [gsky, cmin, cmax], [gsky, cmin, cmax]
        up[i] += h
        dn[i] -= h
        g_fd.append((f_of(*up) - f_of(*dn)) / (2.0 * h))
    return g_ad, g_jvp, tuple(g_fd)


class TestT9ClipGradientDomains:
    @pytest.mark.parametrize("v1,v2,expect", [
        (0.0, 1.0, (1.0, 0.0, 0.0)),      # in band: dF/dgsky = v2 (model-I grad)
        (-1.0, 0.1, (0.0, 1.1, 0.0)),     # lower-clipped: dF/dc_min = v_rel
        (2.9, 3.0, (0.0, 0.0, 0.1)),      # upper-clipped: dF/dc_max = v_rel
        (1.0, 0.0, (0.0, -1.0, 0.0)),     # negative v_rel, upper clip in force
    ], ids=["in-band", "lower-clipped", "upper-clipped", "neg-vrel"])
    def test_clip_force_gradients(self, v1, v2, expect):
        gsky = 2.0  # keeps the samples strictly off the clip boundaries
        g_ad, g_jvp, g_fd = clip_unit_grads(gsky, P.c_min, P.c_max, v1, v2)
        print(f"[D3] T9 ({v1}, {v2}): expect {expect}")
        print(f"     AD  = {tuple(round(x, 9) for x in g_ad)}")
        print(f"     JVP = {tuple(round(x, 9) for x in g_jvp)}")
        print(f"     FD  = {tuple(round(x, 9) for x in g_fd)}")
        assert max(abs(a - e) for a, e in zip(g_ad, expect)) < 1e-9
        assert max(abs(a - e) for a, e in zip(g_jvp, expect)) < 1e-9
        assert max(abs(a - e) for a, e in zip(g_fd, expect)) < 1e-5

    def test_mixed_domain_activation_record(self):
        """Record the on-boundary behaviour of the gate scenario (T9 record)."""
        res = run_semi(P, ROAD, t_end=N_STEPS * DT)
        active = int(res.mask.sum())
        print(f"[D3] T9 mixed-domain record: active steps = {active}/{N_STEPS} "
              f"(bump, default bounds c in [{P.c_min:.4f}, {P.c_max:.4f}], "
              f"gamma_sky = {P.gamma_sky:.4f})")
        assert active > 0       # mixed domain: both branches exercised
        assert active < N_STEPS


# ---------------------------------------------------------------------------
# T13: full-loop three-way gradient validation (JVP vs reverse vs FD)
# ---------------------------------------------------------------------------
PARAM_FIELDS = {
    "gamma_sky": model2.p_gsky2,
    "c_min": model2.p_cmin2,
    "c_max": model2.p_cmax2,
}


def j_forward(**overrides) -> tuple:
    """J_acc (numpy RMS) and the mask channel of a plain forward run."""
    res = run_semi(SemiActiveParams(**overrides), ROAD, t_end=N_STEPS * DT)
    return math.sqrt(np.mean(res.a2 ** 2)), res.mask


class TestT13ThreeWayGradient:
    @pytest.mark.parametrize("param", ["gamma_sky", "c_min", "c_max"])
    def test_three_way_full_loop(self, param):
        # h = 1e-4: an h = 1e-3 window can straddle RK4-stage-level clip
        # switches (whole-step mask equality does not cover the stages),
        # which biases the central FD by O(h) while AD stays exact; at 1e-4
        # the window is kink-free and FD converges to AD (empirical check).
        h = 1.0e-4
        j0, m0 = j_forward()
        up = {param: getattr(P, param) + h}
        dn = {param: getattr(P, param) - h}
        jp, mp = j_forward(**up)
        jm, mm = j_forward(**dn)
        grad_fd = (jp - jm) / (2.0 * h)
        stable = np.array_equal(m0, mp) and np.array_equal(m0, mm)
        print(f"[D3] T13 dJ/d{param}: active-set stable over +/-{h:.0e}: "
              f"{stable} ({int(m0.sum())}/{N_STEPS} active steps)")

        # reverse (adstack) through the full 1000-step loop
        model2.set_params2(P.mu, P.omega_t_sq, P.omega_s_sq, P.gamma_t,
                           P.gamma_sky, P.c_min, P.c_max)
        model2.set_state_hist2(0.0, 0.0, 0.0, 0.0)
        model2.clear_j2()
        with qd.ad.Tape(loss=model2.j_val2):
            model2.rollout_semi(N_STEPS, DT, *ROAD_ARGS)
            model2.reduce_j2(N_STEPS)
            model2.finalize_j2(N_STEPS)
        j_kernel = model2.j_val2[None]
        grad_ad = PARAM_FIELDS[param].grad[None]

        # forward JVP (dual path)
        model2.set_state_hist2(0.0, 0.0, 0.0, 0.0)
        model2.clear_j2()
        with qd.ad.FwdMode(loss=model2.j_val2, param=PARAM_FIELDS[param]):
            model2.rollout_semi(N_STEPS, DT, *ROAD_ARGS)
            model2.reduce_j2(N_STEPS)
            model2.finalize_j2(N_STEPS)
        grad_jvp = model2.j_val2.dual[None]

        j_numpy = math.sqrt(
            np.mean(model2.acc_rec2.to_numpy()[:N_STEPS] ** 2))
        rel = lambda a, b: abs(a - b) / max(abs(b), 1e-300)  # noqa: E731
        print(f"[D3] T13 dJ/d{param}: J(kernel) = {j_kernel:.12e} "
              f"J(numpy) = {j_numpy:.12e}")
        print(f"     AD = {grad_ad:.10e}, JVP = {grad_jvp:.10e}, "
              f"FD = {grad_fd:.10e}")
        print(f"     rel AD/FD = {rel(grad_ad, grad_fd):.3e}, "
              f"JVP/FD = {rel(grad_jvp, grad_fd):.3e}, "
              f"AD/JVP = {rel(grad_ad, grad_jvp):.3e}")
        assert stable                          # smoothness precondition of FD
        assert abs(j_kernel / j_numpy - 1.0) < 1e-12
        assert rel(grad_ad, grad_fd) < 1e-5
        assert rel(grad_jvp, grad_fd) < 1e-5
        assert rel(grad_ad, grad_jvp) < 1e-5


# ---------------------------------------------------------------------------
# ON-OFF branch: forward semantics + explicit zero-gradient marking
# ---------------------------------------------------------------------------
class TestOnOffBranch:
    def test_forward_switching_law(self):
        res = run_onoff(P, ROAD, t_end=1.0)
        v_rel = res.state[:, 3] - res.state[:, 2]
        expect_ct = np.where(res.state[:, 3] * v_rel > 0.0, P.c_max, P.c_min)
        n_hi = int((res.ct == P.c_max).sum())
        print(f"[D3] ON-OFF law: c(t) matches v2*(v2-v1) sign at all "
              f"{len(res.ct)} steps; high-damping steps = {n_hi}")
        assert np.array_equal(res.ct, expect_ct)   # exact switch law
        assert 0 < n_hi < len(res.ct)              # both states exercised

    def test_zero_gradient_marking(self):
        """Inside a Tape the ON-OFF rollout contributes exact 0.0 gradients."""
        model2.set_params2(P.mu, P.omega_t_sq, P.omega_s_sq, P.gamma_t,
                          P.gamma_sky, P.c_min, P.c_max)
        model2.set_state_hist2(0.0, 0.0, 0.0, 0.0)
        model2.clear_j2()
        with qd.ad.Tape(loss=model2.j_val2):
            _onoff_rollout(N_STEPS, DT, *ROAD_ARGS)
            model2.reduce_j2(N_STEPS)
            model2.finalize_j2(N_STEPS)
        grads = (model2.p_gsky2.grad[None], model2.p_cmin2.grad[None],
                 model2.p_cmax2.grad[None])
        print(f"[D3] ON-OFF in Tape: J = {model2.j_val2[None]:.6e}, "
              f"grads (gsky, cmin, cmax) = {grads}")
        assert model2.j_val2[None] > 0.0          # forward ran for real
        assert grads == (0.0, 0.0, 0.0)           # loud structural zero


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main(["-v", "-s", __file__]))
