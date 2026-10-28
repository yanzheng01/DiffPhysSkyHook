"""E2 adjoint gradient tests for the force-command channel (Gate: T16).

Stage E2 of docs/plan_b_force_input_dev_test_plan.md: the whole (T,) adjoint
dJ/df_cmd comes out of ONE reverse Tape pass (plan B decision 3); FwdMode JVP
with FULL-LENGTH seeds (D4 rule) and central differences are the two
verification routes.

Anchors:
  T16 three-way adjoint validation:
    - reverse Tape: one backward pass yields the full (T,) adjoint vector;
      structural assertions: elements n >= n_steps are EXACTLY zero (the
      rollout never reads them), kernel J == numpy RMS.
    - FwdMode JVP: seed = e_i over the FULL (T_MAX,) field, one direction per
      launch, spot-checked against the reverse adjoint.
    - central FD: h = 1e-4 at deterministically sampled steps, each with the
      ACTIVE-SET STABILITY precondition (mask arrays of the +/-h runs
      bit-identical to the baseline -- D3 anchor). In-band samples must match
      to rel < 1e-5; out-of-band (clip-truncated) samples have adjoint exactly
      0 and FD exactly 0 (the +/-h commands stay on the same side of the
      window, so F_d -- and hence J -- is unchanged).

Adjoint statistics (max / min / nonzero count) are printed for every test --
they are a plan B deliverable (section 7) for the co-design signal-quality
assessment. Zero adjoint at a step means the command is clip-truncated there
(dF_d/df_cmd = 0, subgradient semantics).
"""

import math

import numpy as np
import pytest
import quadrants as qd

from skyhook import model2
from skyhook.params import SemiActiveParams
from skyhook.road import BumpRoad
from skyhook.sim2 import run_semi_cmd

P = SemiActiveParams()
DT = 1.0e-3
N_STEPS = 1000
ROAD = BumpRoad(amp=0.05, t0=0.05, width=0.1)
ROAD_ARGS = ROAD.kernel_args()
SEED = 20261028


def baseline_command() -> np.ndarray:
    """Deterministic mixed-domain command series (in/out of band)."""
    return np.random.default_rng(SEED).uniform(-1.0, 1.0, N_STEPS)


def setup_chain(fc: np.ndarray):
    """Launch sequence prefix: params, IC, command writes, J clear."""
    model2.set_params2(P.mu, P.omega_t_sq, P.omega_s_sq, P.gamma_t,
                       P.gamma_sky, P.c_min, P.c_max)
    model2.set_state_hist2(0.0, 0.0, 0.0, 0.0)
    for n in range(len(fc)):
        model2.set_fcmd_elem(n, float(fc[n]))
    model2.clear_j2()


def j_chain(n_steps: int):
    """Differentiable J chain: rollout -> reduce -> finalize."""
    model2.rollout_cmd_semi(n_steps, DT, *ROAD_ARGS)
    model2.reduce_j2(n_steps)
    model2.finalize_j2(n_steps)


def reverse_adjoint(fc: np.ndarray) -> tuple:
    """(J_kernel, adjoint (T_MAX,) array) from one reverse Tape pass."""
    setup_chain(fc)
    with qd.ad.Tape(loss=model2.j_val2):
        j_chain(len(fc))
    return model2.j_val2[None], model2.f_cmd.grad.to_numpy()


def j_numpy(fc: np.ndarray) -> tuple:
    """Plain-forward numpy J and the mask channel."""
    res = run_semi_cmd(P, fc, ROAD, t_end=len(fc) * DT)
    return math.sqrt(np.mean(res.a2 ** 2)), res.mask


class TestT16AdjointThreeWay:
    def test_reverse_adjoint_structure(self):
        """One backward pass: whole (T,) adjoint; tail exactly zero."""
        fc = baseline_command()
        j_kernel, g = reverse_adjoint(fc)
        j_np, _ = j_numpy(fc)
        g_in, tail = g[:N_STEPS], g[N_STEPS:N_STEPS + 64]
        print(f"[E2] T16 reverse: J(kernel) = {j_kernel:.12e} "
              f"J(numpy) = {j_np:.12e}")
        print(f"[E2] T16 adjoint dJ/df_cmd summary: max = {g_in.max():.6e}, "
              f"min = {g_in.min():.6e}, max|g| = {np.abs(g_in).max():.6e}, "
              f"nonzero = {np.count_nonzero(g_in)}/{N_STEPS} "
              f"(zeros = clip-truncated steps), tail max|g| = "
              f"{np.abs(tail).max():.3e}")
        assert abs(j_kernel / j_np - 1.0) < 1e-12
        assert np.count_nonzero(g_in) > 0        # non-trivial adjoint
        assert np.count_nonzero(tail) == 0       # exact structural zero
        assert np.abs(g_in).max() > 0.0

    def test_fd_spot_checks(self):
        """Central FD at deterministic samples; active-set stability first."""
        h = 1.0e-4
        fc = baseline_command()
        _, g = reverse_adjoint(fc)
        g_in = g[:N_STEPS]
        j0, mask0 = j_numpy(fc)
        checked, n_zero = 0, 0
        for i in range(0, N_STEPS, N_STEPS // 24):
            if checked >= 12:
                break
            fcp, fcm = fc.copy(), fc.copy()
            fcp[i] += h
            fcm[i] -= h
            jp, mp = j_numpy(fcp)
            jm, mm = j_numpy(fcm)
            stable = np.array_equal(mask0, mp) and np.array_equal(mask0, mm)
            assert stable, f"active-set moved at step {i}: FD window not smooth"
            fd = (jp - jm) / (2.0 * h)
            if g_in[i] == 0.0:
                assert fd == 0.0   # truncated step: J is locally constant
                n_zero += 1
            else:
                rel = abs(g_in[i] - fd) / abs(g_in[i])
                assert rel < 1e-5, f"step {i}: adjoint {g_in[i]:.6e} vs FD {fd:.6e}"
            checked += 1
        print(f"[E2] T16 FD: {checked} mask-stable samples checked "
              f"({n_zero} clip-truncated zeros exact), h = {h:.0e}, "
              f"all rel < 1e-5 (nonzero) / exact 0 (truncated)")

    def test_jvp_spot_checks(self):
        """FwdMode JVP with full-length e_i seeds vs the reverse adjoint."""
        fc = baseline_command()
        _, g = reverse_adjoint(fc)
        g_in = g[:N_STEPS]
        top = np.argsort(-np.abs(g_in))[:3]   # strongest 3 directions
        for i in top:
            setup_chain(fc)
            seed = [0.0] * model2.T_MAX
            seed[int(i)] = 1.0
            with qd.ad.FwdMode(loss=model2.j_val2, param=model2.f_cmd,
                               seed=seed):
                j_chain(N_STEPS)
            g_jvp = model2.j_val2.dual[None]
            rel = abs(g_jvp - g_in[i]) / max(abs(g_in[i]), 1e-300)
            print(f"[E2] T16 JVP i={int(i)}: reverse = {g_in[i]:.10e}, "
                  f"JVP = {g_jvp:.10e}, rel = {rel:.3e}")
            assert rel < 1e-9


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main(["-v", "-s", __file__]))
