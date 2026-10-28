"""E3 open-loop force-sequence optimization tests (Gate: T17).

Stage E3 of docs/plan_b_force_input_dev_test_plan.md: projected-gradient
DESCENT WITHOUT projection (plan B decision 2 -- the model-side clip makes
every command realizable) on the (n_steps,) command series, gradients from
one reverse Tape pass per iterate (decision 3, validated by T16).

Anchors:
  T17a zero-start plateau (structural): f_cmd == 0 makes F_d saturate on the
      window bound c_min*|v_rel| at every moving step (T14) and the window is
      zero-width at every v_rel == 0 step, so dF_d/df_cmd = 0 almost
      everywhere and the whole adjoint is EXACTLY zero: the zero command is
      a clip fixed point and J is locally constant there. optimize_force must
      stop loudly with stopped = "zero_gradient" -- the honest statement that
      gradient descent cannot start on the plateau (any open-loop controller
      starts from a nonzero command anyway).
  T17b optimization smoke from a deterministic nonzero start: J strictly
      monotone decreasing (Armijo sufficient decrease, no projection), with
      the command-out-of-band (realizability) statistics recorded.
  T17c sparse directional derivative FD: normalized sparse -g direction
      (support = top-8 |adjoint| entries), central differences with h = 1e-4
      and the ACTIVE-SET STABILITY precondition (mask arrays of the +/-h runs
      bit-identical, D3 anchor), at both the optimization start and the final
      iterate. Deterministic seeds; no retry needed (stability verified).
"""

import math

import numpy as np
import pytest

from skyhook.opt import force_adjoint, optimize_force, rollout_force_j
from skyhook.params import SemiActiveParams
from skyhook.road import BumpRoad
from skyhook.sim2 import run_semi_cmd

P = SemiActiveParams()
DT = 1.0e-3
N_STEPS = 1000
ROAD = BumpRoad(amp=0.05, t0=0.05, width=0.1)
X0 = (0.0, 0.0, 0.0, 0.0)
FC0 = np.random.default_rng(42).uniform(-0.2, 0.2, N_STEPS)


class TestT17ForceOptimization:
    def test_zero_start_plateau(self):
        """f_cmd == 0: adjoint exactly zero, optimizer stops loudly."""
        res = optimize_force(P, ROAD, x0=X0, n_steps=N_STEPS, dt=DT,
                             f_cmd0=np.zeros(N_STEPS), n_iters=5)
        j_ref, _ = force_adjoint(np.zeros(N_STEPS), P, ROAD, X0,
                                 N_STEPS, DT)
        print(f"[E3] T17a zero start: stopped = '{res.stopped}', "
              f"J = {res.j[0]:.12e} (= passive c_min anchor {j_ref:.12e}), "
              f"|grads[0]|max = {np.nanmax(np.abs(res.grads[0])):.3e}")
        assert res.stopped == "zero_gradient"
        assert len(res.j) == 1                    # no iterate accepted
        assert res.j[0] == j_ref
        assert np.count_nonzero(res.grads[0]) == 0   # exact structural zero

    def test_optimization_smoke(self):
        """Nonzero deterministic start: strict monotone decrease + stats."""
        res = optimize_force(P, ROAD, x0=X0, n_steps=N_STEPS, dt=DT,
                             f_cmd0=FC0, n_iters=15, alpha0=10.0)
        drops = np.diff(res.j)
        res_final = run_semi_cmd(P, res.fc[-1], ROAD, x0=X0,
                                 t_end=N_STEPS * DT)
        n_out = int(res_final.mask.sum())
        print(f"[E3] T17b smoke: stopped = '{res.stopped}', "
              f"J: {res.j[0]:.6f} -> {res.j[-1]:.6f} "
              f"(-{100.0 * (1.0 - res.j[-1] / res.j[0]):.2f}%), "
              f"min step drop = {drops.min():.3e} (> 0: strictly monotone), "
              f"alphas = {res.alphas[0]}..{res.alphas[-1]}, "
              f"final out-of-band = {n_out}/{N_STEPS}, "
              f"|fc|max = {np.abs(res.fc[-1]).max():.4f}")
        assert res.stopped == "max_iter"
        assert np.all(drops < 0.0)                # strictly monotone segment
        assert res.j[-1] < res.j[0]
        assert 0 < n_out < N_STEPS                # realizability statistics

    def test_sparse_direction_fd(self):
        """Normalized sparse -g directions: AD vs central FD, mask stable."""
        h = 1.0e-4
        for tag, fc_base in [("start", FC0), ("final", None)]:
            if fc_base is None:
                fc_base = optimize_force(P, ROAD, x0=X0, n_steps=N_STEPS,
                                         dt=DT, f_cmd0=FC0, n_iters=3,
                                         alpha0=10.0).fc[-1]
            j_b, g = force_adjoint(fc_base, P, ROAD, X0, N_STEPS, DT)
            support = np.argsort(-np.abs(g))[:8]
            d = np.zeros(N_STEPS)
            d[support] = -g[support]
            d[support] /= np.linalg.norm(g[support])
            dd_ad = float(np.dot(g, d))
            m0 = run_semi_cmd(P, fc_base, ROAD, x0=X0,
                              t_end=N_STEPS * DT).mask
            fcp = fc_base + h * d
            fcm = fc_base - h * d
            jp = rollout_force_j(fcp, P, ROAD, X0, N_STEPS, DT)
            jm = rollout_force_j(fcm, P, ROAD, X0, N_STEPS, DT)
            dd_fd = (jp - jm) / (2.0 * h)
            mp = run_semi_cmd(P, fcp, ROAD, x0=X0, t_end=N_STEPS * DT).mask
            mm = run_semi_cmd(P, fcm, ROAD, x0=X0, t_end=N_STEPS * DT).mask
            stable = np.array_equal(m0, mp) and np.array_equal(m0, mm)
            rel = abs(dd_ad - dd_fd) / max(abs(dd_ad), 1e-300)
            print(f"[E3] T17c {tag}: J = {j_b:.6f}, dd_ad = {dd_ad:+.8e}, "
                  f"dd_fd = {dd_fd:+.8e}, rel = {rel:.2e}, "
                  f"active-set stable = {stable}")
            assert stable
            assert rel < 1e-5


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main(["-v", "-s", __file__]))
