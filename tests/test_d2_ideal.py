"""D2 model-I (ideal skyhook) tests (Gate: T4, T5, T6).

Stage D2 of docs/plan_a_dev_test_plan.md: the passive quarter-car gains the
single ideal-skyhook term -gamma_sky * v2 (docs/plan_a_skyhook_model.md
section 3). The term is exclusive to the sprung-mass row: the virtual damper
reacts against the inertial frame and transmits no force to the unsprung mass.

Anchors:
  T4  body-mode decay rate sigma measured from an exact single-mode free
      response (initial state = damped QEP eigenvector, so the record is a
      pure Re(A e^{s1 t}) and the LSQ decay fit is bias-free):
      (a) pure skyhook (gamma_s = gamma_t = 0) against the plan formula
          zeta_eff = (gamma_s + gamma_sky)/(2 omega_s) via
          zeta_meas = sigma_meas / omega_s; tolerance 5%;
      (b) default suspension damping (gamma_s > 0) against the exact damped
          QEP root; tolerance 2%.
      The QEP-vs-formula gap itself is asserted < 1%: comparing modal
      *frequencies* would carry a ~7.6% bias (modal mass 1 + mu/r^2), so the
      decay-rate comparison sigma = zeta_eff * omega_s is the honest anchor.
  T5  reverse-mode AD gradient dJ/d(gamma_sky) through the full 1000-step
      time loop (qd.ad.Tape around rk4_step + reduce_j + finalize_j) vs a
      central finite difference on the forward simulator; tolerance 1e-5
      relative. Also asserts kernel J == numpy RMS(acc_rec) to 1e-12.
  T6  structural exclusivity: toggling gamma_sky leaves row 1 (v1dot)
      bit-identical and shifts row 2 (v2dot) by exactly -gamma_sky * v2.
"""

import math

import numpy as np
import pytest
import quadrants as qd

from analytic import body_mode_ic, fit_mode_sigma, qep_damped_roots
from skyhook import model
from skyhook.params import PassiveParams
from skyhook.road import BumpRoad, ROAD_ZERO, ZeroRoad
from skyhook.sim import run_passive

P = PassiveParams()
DT = 1.0e-3
GSKY = 2.0 * 0.30 * P.omega_s  # zeta_sky = 0.30 anchor


# body_mode_ic lives in analytic.py (shared with D4 T10); its eigenvector
# formula was corrected in D4: phi1's denominator is gamma_s*s + omega_s^2
# (row-2 eigenrelation), not omega_s^2.


# ---------------------------------------------------------------------------
# T4: body-mode effective damping ratio
# ---------------------------------------------------------------------------
class TestT4EffectiveDamping:
    @staticmethod
    def _measure(p: PassiveParams):
        x0, s = body_mode_ic(p)
        res = run_passive(p, ZeroRoad(), x0=x0, t_end=6.0, dt=DT)
        sigma0 = p.zeta_eff * p.omega_s
        sigma = fit_mode_sigma(res.t, res.state[:, 1], sigma0, p.omega_s)
        return sigma, s

    def test_pure_skyhook_plan_formula(self):
        p = PassiveParams(gamma_t=0.0, gamma_s=0.0, gamma_sky=GSKY)
        sigma, s = self._measure(p)
        zeta_meas = sigma / p.omega_s
        gap = abs((-s.real) / (p.zeta_eff * p.omega_s) - 1.0)
        print(f"[D2] T4 pure skyhook: sigma={sigma:.6f} 1/s, zeta_eff meas="
              f"{zeta_meas:.6f} vs formula {p.zeta_eff:.6f}, rel err="
              f"{abs(zeta_meas / p.zeta_eff - 1.0):.3e}; "
              f"QEP-vs-formula gap={gap:.3e}")
        assert gap < 0.01                      # both anchors agree first
        assert abs(zeta_meas / p.zeta_eff - 1.0) < 0.05

    def test_with_suspension_damping_qep(self):
        p = PassiveParams(gamma_sky=GSKY)
        sigma, s = self._measure(p)
        rel = abs(sigma / (-s.real) - 1.0)
        print(f"[D2] T4 damped case: sigma meas={sigma:.6f} vs QEP root "
              f"{-s.real:.6f} 1/s, rel err={rel:.3e}")
        assert rel < 0.02


# ---------------------------------------------------------------------------
# T5: dJ/d(gamma_sky) reverse-mode AD vs central finite difference
# ---------------------------------------------------------------------------
class TestT5SkyhookGradient:
    def test_dj_dgamma_sky_ad_vs_fd(self):
        p = PassiveParams(gamma_sky=5.0)
        road = BumpRoad(amp=0.05, t0=0.05, width=0.1)
        n_steps = 1000
        kind, amp, w, t0, width, ramp_t = road.kernel_args()

        # tape setup outside the recording context (state/params/accumulator);
        # the rollout runs as ONE kernel with per-step state slots (x_hist,
        # written once, never overwritten) so reverse mode always reads
        # intact primal values
        model.set_params(p.mu, p.omega_t_sq, p.omega_s_sq,
                         p.gamma_t, p.gamma_s, p.gamma_sky)
        model.set_state_hist(0.0, 0.0, 0.0, 0.0)
        model.clear_j()
        with qd.ad.Tape(loss=model.j_val):
            model.rollout_record(n_steps, DT, kind, amp, w, t0, width, ramp_t)
            model.reduce_j(n_steps)
            model.finalize_j(n_steps)
        j_kernel = model.j_val[None]
        grad_ad = model.p_gsky.grad[None]

        a2 = model.acc_rec.to_numpy()[:n_steps]
        j_numpy = math.sqrt(np.mean(a2 * a2))

        h = 1.0e-3

        def j_of(gsky: float) -> float:
            res = run_passive(PassiveParams(gamma_sky=gsky), road,
                              t_end=n_steps * DT, dt=DT)
            return math.sqrt(np.mean(res.a2 ** 2))

        grad_fd = (j_of(p.gamma_sky + h) - j_of(p.gamma_sky - h)) / (2.0 * h)
        rel = abs(grad_ad - grad_fd) / abs(grad_fd)
        print(f"[D2] T5: J(kernel)={j_kernel:.12e}, J(numpy)={j_numpy:.12e}, "
              f"dJ/dgsky AD={grad_ad:.10e}, FD={grad_fd:.10e}, rel err={rel:.3e}")
        assert abs(j_kernel / j_numpy - 1.0) < 1e-12
        assert math.isfinite(grad_ad)
        assert rel < 1e-5


# ---------------------------------------------------------------------------
# T6: the skyhook term is exclusive to the sprung-mass row
# ---------------------------------------------------------------------------
class TestT6Exclusivity:
    @pytest.mark.parametrize("state,gsky", [
        ((0.01, -0.02, 0.3, -0.7), 3.7),
        ((0.0, 0.0, 1.0, 1.0), 0.5),
        ((-0.05, 0.05, 0.0, 2.0), 10.0),
    ], ids=["mixed", "shear-free", "high-v2"])
    def test_skyhook_term_exclusive_to_row2(self, state, gsky):
        base = PassiveParams()  # gamma_sky = 0
        prm = (base.mu, base.omega_t_sq, base.omega_s_sq,
               base.gamma_t, base.gamma_s)
        probe_args = (0.0, ROAD_ZERO, 0.0, 0.0, 0.0, 0.1, 0.0)

        model.set_params(*prm, 0.0)
        model.set_state(*state)
        model.probe_rhs(*probe_args)
        a1_off = model.a1_probe[None]
        a2_off = model.a2_probe[None]

        model.set_params(*prm, gsky)
        model.probe_rhs(*probe_args)
        a1_on = model.a1_probe[None]
        a2_on = model.a2_probe[None]

        expect = -gsky * state[3]
        diff = a2_on - a2_off
        print(f"[D2] T6: a1 bit-identical={a1_on == a1_off}, a2 diff="
              f"{diff:.12e} vs -gsky*v2={expect:.12e}")
        assert a1_on == a1_off  # bit-exact: row 1 never sees gamma_sky
        assert abs(diff - expect) <= 1e-12 * max(1.0, abs(expect))
