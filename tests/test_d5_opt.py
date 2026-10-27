"""D5 optimization-loop tests (Gate: T12).

Stage D5 of docs/plan_a_dev_test_plan.md: single-variable gamma_sky descent
on J_acc (single bump), then the full trio (gamma_sky, c_min, c_max). The
gradient route is the DEFAULT FORWARD JVP path (one FwdMode sweep per
parameter, skyhook/opt.rollout_grad); the optimizer is projected gradient
descent with Armijo backtracking.

T12 pass criteria "monotone segment + correct direction":
  - monotone segment: the iterate record contains a run of >= 10 strictly
    decreasing J values (observed: 20/20 and 25/25, every accepted step
    strictly decreases by construction of the Armijo rule);
  - correct direction: the JVP directional derivative along the normalized
    steepest-descent direction matches a central finite difference
    (h = 1e-4, active-set stability asserted first -- the D3 finding that
    makes the truncated-subgradient AD comparable to FD), is negative, and
    the optimizer realizes the predicted decrease (J_end < J_start by > 1%).
"""

import numpy as np

from skyhook.opt import PARAM_ORDER, optimize, rollout_j
from skyhook.params import SemiActiveParams
from skyhook.road import BumpRoad
from skyhook.sim2 import run_semi

ROAD = BumpRoad(amp=0.05, t0=0.05, width=0.1)
N_STEPS = 1000
DT = 1.0e-3
X0 = (0.0, 0.0, 0.0, 0.0)
H = 1.0e-4
P0 = SemiActiveParams()


def _longest_strict_decrease(j: np.ndarray) -> int:
    best = run = 0
    for d in j[1:] - j[:-1]:
        run = run + 1 if d < 0.0 else 0
        best = max(best, run)
    return best


def _directional_check(th, g, label):
    """FD vs JVP directional derivative along normalized -g (+ stability)."""
    gnorm = float(np.linalg.norm(g))
    d = -g / gnorm
    up = SemiActiveParams(gamma_sky=th[0] + H * d[0], c_min=th[1] + H * d[1],
                          c_max=th[2] + H * d[2])
    dn = SemiActiveParams(gamma_sky=th[0] - H * d[0], c_min=th[1] - H * d[1],
                          c_max=th[2] - H * d[2])
    base = SemiActiveParams(gamma_sky=th[0], c_min=th[1], c_max=th[2])
    r0 = run_semi(base, ROAD, t_end=N_STEPS * DT)
    rp = run_semi(up, ROAD, t_end=N_STEPS * DT)
    rm = run_semi(dn, ROAD, t_end=N_STEPS * DT)
    stable = (np.array_equal(r0.mask, rp.mask)
              and np.array_equal(r0.mask, rm.mask))
    dd_fd = (rollout_j(up, ROAD, X0, N_STEPS, DT)
             - rollout_j(dn, ROAD, X0, N_STEPS, DT)) / (2.0 * H)
    dd_ad = float(np.dot(g, d))
    rel = abs(dd_fd - dd_ad) / max(abs(dd_ad), 1e-300)
    print(f"[D5] {label}: dir deriv FD={dd_fd:.10e} JVP={dd_ad:.10e} "
          f"rel={rel:.3e}, |g|={gnorm:.3e}, "
          f"active-set stable={stable} ({int(r0.mask.sum())}/{N_STEPS})")
    return dd_ad, rel, stable


class TestT12Optimization:
    def test_single_variable_gamma_sky_descent(self):
        """Stage 1: gamma_sky-only PGD from a low start (T13 bump scenario)."""
        start = SemiActiveParams(gamma_sky=0.25 * P0.gamma_sky)
        res = optimize(start, ROAD, x0=X0, n_steps=N_STEPS, dt=DT,
                       which=("gamma_sky",), n_iters=20)

        mono = _longest_strict_decrease(res.j)
        rel_drop = float(res.j[-1] / res.j[0] - 1.0)
        print(f"[D5] T12 single-var: stopped={res.stopped}, "
              f"steps={res.n_steps}, evals={res.n_evals}, "
              f"gamma_sky {res.th[0, 0]:.4f} -> {res.th[-1, 0]:.4f}, "
              f"J {res.j[0]:.8f} -> {res.j[-1]:.8f} (rel {rel_drop:.3e}), "
              f"longest strict decrease={mono}")

        assert res.n_steps >= 10                     # a real descent run
        assert mono >= 10                            # the monotone segment
        assert rel_drop < -0.01                      # realized decrease > 1%
        assert np.all(res.th[:, 0] >= 0.0)           # projection respected

        # correct direction at the start iterate: JVP vs FD along -g/g
        dd_ad, rel, stable = _directional_check(
            res.th[0], res.grads[0], "T12 single-var dir@k0")
        assert stable                               # FD validity precondition
        assert dd_ad < 0.0                          # descent direction
        assert rel < 1.0e-5

    def test_three_variable_trio_descent(self):
        """Stage 2: the full (gamma_sky, c_min, c_max) trio, mixed start."""
        start = SemiActiveParams(gamma_sky=0.5 * P0.gamma_sky,
                                 c_min=2.0 * P0.c_min,
                                 c_max=0.5 * P0.c_max)
        res = optimize(start, ROAD, x0=X0, n_steps=N_STEPS, dt=DT,
                       which=PARAM_ORDER, n_iters=25)

        mono = _longest_strict_decrease(res.j)
        rel_drop = float(res.j[-1] / res.j[0] - 1.0)
        print(f"[D5] T12 trio: stopped={res.stopped}, "
              f"steps={res.n_steps}, evals={res.n_evals}, "
              f"th {np.round(res.th[0], 4)} -> {np.round(res.th[-1], 4)}, "
              f"J {res.j[0]:.8f} -> {res.j[-1]:.8f} (rel {rel_drop:.3e}), "
              f"longest strict decrease={mono}")

        assert res.n_steps >= 10
        assert mono >= 10
        assert rel_drop < -0.01
        # projection respected on every iterate
        assert np.all(res.th[:, 0] >= 0.0)
        assert np.all(res.th[:, 1] >= 0.0)
        assert np.all(res.th[:, 2] > res.th[:, 1])

        # correct direction at the start iterate (all three directions live)
        g = res.grads[0]
        assert np.count_nonzero(g) == 3             # all trio gradients flow
        dd_ad, rel, stable = _directional_check(res.th[0], g, "T12 trio dir@k0")
        assert stable
        assert dd_ad < 0.0
        assert rel < 1.0e-5
