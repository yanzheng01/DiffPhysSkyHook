"""D4 batch + integrator-comparison tests (Gate: T10, T11, T13 batch leg).

Stage D4 of docs/plan_a_dev_test_plan.md: batch dimension and integrator
comparison. Three anchors:

  T10 integrator convergence order on the smooth linear model I (passive,
      default gains, zero road): body-mode IC (exact damped QEP eigenvector
      via analytic.body_mode_ic), so x2(t) = amp e^{sigma t} cos(w t) exactly.
      Observed order p = log2(err_k / err_{k+1}) over dt halvings
      [20, 10, 5, 2.5, 1.25] ms at T = 1 s:
      RK4 (production, dt = 1 ms locked elsewhere) ~ 4,
      semi-implicit Euler (companion only) ~ 1, tolerance +/- 0.3.
  T11 batch-vs-expand: run_semi_batch (B = 8, mixed params / roads / ICs)
      vs B independent run_semi rollouts, element-wise on state / a2 / fd /
      mask / ft / j / final_state. Expected (and observed) bit-exact: the
      per-element step body is identical arithmetic.
  T13 batch three-way gradient: dJ_mean / dgamma_sky[b] through the full
      batched 1000-step loop, reverse Tape (full gradient vector) vs
      FwdMode JVP (seed = e_i over the FULL (B_MAX,) field -- the seed list
      must have the field's full length, not B) vs central FD (h = 1e-4,
      active-set stability asserted, D3 finding). Tolerance 1e-5 relative.

Numerical caveat (D4 probe finding, encoded in the fixtures): a batch
element with J = 0 (flat road + zero IC) makes dual(j_val_b) = sqrt'(0) =
0/0 = nan, which poisons j_mean's dual chain. Flat-road elements therefore
get a nonzero IC (free decay).
"""

import math

import numpy as np
import pytest
import quadrants as qd

from analytic import body_mode_ic
from d4_fixture import B, DT, N_STEPS, PARAMS, ROADS, X0S, perturbed
from skyhook import model2b
from skyhook.params import PassiveParams, SemiActiveParams
from skyhook.road import ZeroRoad
from skyhook.sim import run_passive, run_passive_euler
from skyhook.sim2 import run_semi
from skyhook.sim2b import _setup, run_j_chain, run_semi_batch


# ---------------------------------------------------------------------------
# T10: integrator convergence order (model I, exact body-mode response)
# ---------------------------------------------------------------------------
class TestT10IntegratorOrder:
    DTS = (0.02, 0.01, 0.005, 0.0025, 0.00125)

    @staticmethod
    def _max_err(dt: float, runner) -> float:
        p = PassiveParams()
        x0, s = body_mode_ic(p)
        res = runner(p, ZeroRoad(), x0=x0, t_end=1.0, dt=dt)
        exact = (1.0e-3 * np.exp(s.real * res.t) * np.cos(s.imag * res.t))
        return float(np.max(np.abs(res.state[:, 1] - exact)))

    @pytest.mark.parametrize("runner,target", [
        (run_passive, 4.0),          # RK4, the production integrator
        (run_passive_euler, 1.0),    # semi-implicit Euler companion
    ], ids=["rk4", "euler"])
    def test_observed_order(self, runner, target):
        errs = [self._max_err(dt, runner) for dt in self.DTS]
        orders = [math.log2(errs[k] / errs[k + 1])
                  for k in range(len(errs) - 1)]
        p_obs = float(np.mean(orders))
        print(f"[D4] T10 {runner.__name__}: errs="
              f"{['%.3e' % e for e in errs]}, orders="
              f"{['%.2f' % o for o in orders]}, mean order={p_obs:.3f}")
        assert errs[-1] < errs[0]                # actually converging
        assert abs(p_obs - target) < 0.3


# ---------------------------------------------------------------------------
# T11: batch dimension vs B independent single rollouts
# ---------------------------------------------------------------------------
class TestT11BatchExpand:
    def test_batch_equals_expanded_singles(self):
        batch = run_semi_batch(PARAMS, ROADS, x0s=X0S,
                               t_end=N_STEPS * DT, dt=DT)
        max_diff = 0.0
        for b in range(B):
            single = run_semi(PARAMS[b], ROADS[b], x0=X0S[b],
                              t_end=N_STEPS * DT, dt=DT)
            max_diff = max(
                max_diff,
                np.max(np.abs(batch.state[b] - single.state)),
                np.max(np.abs(batch.a2[b] - single.a2)),
                np.max(np.abs(batch.fd[b] - single.fd)),
                np.max(np.abs(batch.mask[b] - single.mask)),
                np.max(np.abs(batch.ft[b] - single.ft)),
                np.max(np.abs(batch.final_state[b] - single.final_state)),
            )
        j_numpy = np.sqrt(np.mean(batch.a2 ** 2, axis=1))
        j_diff = float(np.max(np.abs(batch.j - j_numpy)))
        print(f"[D4] T11: max element-wise diff = {max_diff:.3e}, "
              f"max |j(kernel)-j(numpy)| = {j_diff:.3e}, "
              f"j_mean(kernel)={batch.j_mean:.12e} "
              f"vs numpy {float(np.mean(j_numpy)):.12e}")
        assert max_diff < 1.0e-12
        assert j_diff < 1.0e-12
        assert abs(batch.j_mean / float(np.mean(j_numpy)) - 1.0) < 1.0e-12


# ---------------------------------------------------------------------------
# T13 (batch leg): three-way dJ_mean/dgamma_sky[b] through the batched loop
# ---------------------------------------------------------------------------
class TestT13BatchThreeWayGradient:
    @pytest.mark.parametrize("i", [0, 3, 6], ids=["b0", "b3", "b6"])
    def test_three_way_per_element(self, i):
        h = 1.0e-4  # D3 finding: h=1e-3 straddles stage-level clip switches

        # central FD on the plain forward API (+ mask stability)
        up_params = perturbed(PARAMS, i, +h)
        dn_params = perturbed(PARAMS, i, -h)
        res0 = run_semi_batch(PARAMS, ROADS, x0s=X0S,
                              t_end=N_STEPS * DT, dt=DT)
        res_up = run_semi_batch(up_params, ROADS, x0s=X0S,
                               t_end=N_STEPS * DT, dt=DT)
        res_dn = run_semi_batch(dn_params, ROADS, x0s=X0S,
                                t_end=N_STEPS * DT, dt=DT)
        grad_fd = (res_up.j_mean - res_dn.j_mean) / (2.0 * h)
        stable = (np.array_equal(res0.mask, res_up.mask)
                  and np.array_equal(res0.mask, res_dn.mask))

        # reverse (adstack): full gradient vector from one Tape
        _setup(PARAMS, ROADS, X0S)
        with qd.ad.Tape(loss=model2b.j_mean):
            run_j_chain(N_STEPS, DT, B)
        grad_ad = model2b.pg_b.grad.to_numpy()[i]

        # forward JVP: seed = e_i over the FULL (B_MAX,) parameter field
        seed = [0.0] * model2b.B_MAX
        seed[i] = 1.0
        _setup(PARAMS, ROADS, X0S)
        with qd.ad.FwdMode(loss=model2b.j_mean, param=model2b.pg_b,
                           seed=seed):
            run_j_chain(N_STEPS, DT, B)
        grad_jvp = model2b.j_mean.dual[None]

        rel = lambda a, b: abs(a - b) / max(abs(b), 1e-300)  # noqa: E731
        print(f"[D4] T13 b={i}: AD={grad_ad:.10e} JVP={grad_jvp:.10e} "
              f"FD={grad_fd:.10e}")
        print(f"     rel AD/FD={rel(grad_ad, grad_fd):.3e}, "
              f"JVP/FD={rel(grad_jvp, grad_fd):.3e}, "
              f"AD/JVP={rel(grad_ad, grad_jvp):.3e}, "
              f"active-set stable={stable}")
        assert stable
        assert rel(grad_ad, grad_fd) < 1.0e-5
        assert rel(grad_jvp, grad_fd) < 1.0e-5
        assert rel(grad_ad, grad_jvp) < 1.0e-5

    def test_reverse_gradient_vector_vs_fd(self):
        """The reverse Tape returns the whole (B,) vector in one sweep."""
        h = 1.0e-4
        res0 = run_semi_batch(PARAMS, ROADS, x0s=X0S,
                              t_end=N_STEPS * DT, dt=DT)
        _setup(PARAMS, ROADS, X0S)
        with qd.ad.Tape(loss=model2b.j_mean):
            run_j_chain(N_STEPS, DT, B)
        grad_ad = model2b.pg_b.grad.to_numpy()[:B]

        grad_fd = np.zeros(B)
        for b in range(B):
            ju = run_semi_batch(perturbed(PARAMS, b, +h), ROADS, x0s=X0S,
                               t_end=N_STEPS * DT, dt=DT).j_mean
            jd = run_semi_batch(perturbed(PARAMS, b, -h), ROADS, x0s=X0S,
                                t_end=N_STEPS * DT, dt=DT).j_mean
            grad_fd[b] = (ju - jd) / (2.0 * h)
        rel = np.abs(grad_ad - grad_fd) / np.maximum(np.abs(grad_fd), 1e-300)
        print(f"[D4] T13 vector: rel AD/FD per element = "
              f"{['%.2e' % r for r in rel]}")
        print(f"     grad_ad={np.round(grad_ad, 8)}")
        print(f"     j_mean={res0.j_mean:.12e}")
        assert np.all(np.isfinite(grad_ad))
        assert np.all(rel < 1.0e-5)
