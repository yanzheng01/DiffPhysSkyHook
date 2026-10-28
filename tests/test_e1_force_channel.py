"""E1 force-command channel tests (Gate: T14, T15).

Stage E1 of docs/plan_b_force_input_dev_test_plan.md: an external force
command F_cmd(t) drives the semi-active device; realizability is enforced
model-side by fd_clip_cmd (plan B section 2-2) -- the optimizer never
projects, the mask channel is the command-out-of-band rate.

Anchors:
  T14 command degeneracy: f_cmd == 0 makes F_d the analytic projection of 0
      onto the admissible window, i.e. F_d = c_min * v_rel exactly, so the
      trajectory is the passive c_min damper: vs run_passive(gamma_s=c_min),
      rel RMS difference < 1e-12 (plan A T8a measured ~3e-16 for the same
      class of comparison). Note: for f_cmd == 0 the command sits OUTSIDE
      the window whenever v_rel != 0 (the window has the sign of v_rel), so
      the out-of-band mask is 1 on every moving step -- recorded, expected.
  T15 passivity under any command: for arbitrary / adversarial commands
      (including grossly out-of-band ones) F_d * v_rel >= 0 at every whole
      step -- the clip window always has the sign of v_rel. The realized
      force and the mask channel are asserted BIT-EXACT against the host-side
      recomputation (identical IEEE ops on identical recorded state), and the
      out-of-band statistics are recorded for the co-design diagnostics.
"""

import math

import numpy as np
import pytest

from skyhook.params import PassiveParams, SemiActiveParams
from skyhook.road import BumpRoad, ZeroRoad
from skyhook.sim import run_passive
from skyhook.sim2 import run_semi_cmd

P = SemiActiveParams()
DT = 1.0e-3
N_STEPS = 1000
ROAD = BumpRoad(amp=0.05, t0=0.05, width=0.1)
BIG = 1.0e6 * P.c_max   # grossly out-of-band amplitude


def force_window(state):
    """Admissible force window [lo, hi] at the recorded whole-step states."""
    v_rel = state[:, 3] - state[:, 2]
    lo = np.minimum(P.c_min * v_rel, P.c_max * v_rel)
    hi = np.maximum(P.c_min * v_rel, P.c_max * v_rel)
    return lo, hi, v_rel


def expected_channels(res, fc):
    """Host-side recomputation of (F_d, mask) from the recorded state."""
    lo, hi, v_rel = force_window(res.state)
    fd = np.minimum(np.maximum(fc, lo), hi)
    mask = ((fc < lo) | (fc > hi)).astype(np.float64)
    return fd, mask, v_rel


# ---------------------------------------------------------------------------
# T14: command degeneracy (f_cmd == 0 is the passive c_min damper)
# ---------------------------------------------------------------------------
class TestT14CommandDegeneracy:
    @pytest.mark.parametrize("road,x0", [
        (ROAD, (0.0, 0.0, 0.0, 0.0)),
        (ZeroRoad(), (1.0e-3, 1.2e-3, 0.0, 0.0)),
    ], ids=["bump", "free-decay"])
    def test_zero_command_is_passive_cmin(self, road, x0):
        res_cmd = run_semi_cmd(P, np.zeros(N_STEPS), road, x0=x0, t_end=1.0)
        res_pas = run_passive(PassiveParams(gamma_s=P.c_min, gamma_sky=0.0),
                              road, x0=x0, t_end=1.0)
        _, _, v_rel = force_window(res_cmd.state)
        fd_err = float(np.max(np.abs(res_cmd.fd - P.c_min * v_rel)))
        rms = math.sqrt(np.mean((res_cmd.a2 - res_pas.a2) ** 2))
        rel = rms / math.sqrt(np.mean(res_pas.a2 ** 2))
        state_rms = float(np.sqrt(np.mean((res_cmd.state - res_pas.state) ** 2)))
        n_out = int(res_cmd.mask.sum())
        print(f"[E1] T14: max|F_d - c_min*v_rel| = {fd_err:.3e}, "
              f"RMS(a2 diff) = {rms:.3e}, rel = {rel:.3e}, "
              f"RMS(state diff) = {state_rms:.3e}, "
              f"out-of-band steps = {n_out}/{N_STEPS} "
              f"(zero command violates the window on every moving step)")
        assert fd_err < 1e-14
        assert rel < 1e-12
        assert state_rms < 1e-12
        # zero command is in-band ONLY where the window degenerates to [0, 0]
        in_band = res_cmd.mask == 0.0
        assert np.all(v_rel[in_band] == 0.0)


# ---------------------------------------------------------------------------
# T15: passivity under any command + mask diagnostics
# ---------------------------------------------------------------------------
class TestT15PassivityAnyCommand:
    @pytest.mark.parametrize("seed", [20261028, 7, 12345])
    def test_random_huge_commands(self, seed):
        """Grossly out-of-band random commands: pure saturation regime."""
        rng = np.random.default_rng(seed)
        fc = rng.uniform(-BIG, BIG, N_STEPS)
        res = run_semi_cmd(P, fc, ROAD, t_end=1.0)
        fd_e, mask_e, v_rel = expected_channels(res, fc)
        products = res.fd * v_rel
        print(f"[E1] T15 seed={seed}: min F_d*v_rel = {np.min(products):.3e}, "
              f"out-of-band = {int(res.mask.sum())}/{N_STEPS} "
              f"(bit-exact fd/mask vs host: "
              f"{np.array_equal(res.fd, fd_e) and np.array_equal(res.mask, mask_e)})")
        assert np.min(products) >= -1e-12
        assert np.array_equal(res.fd, fd_e)
        assert np.array_equal(res.mask, mask_e)

    def test_moderate_commands_mixed_domain(self):
        """Window-scale commands: mixed in-band / out-of-band steps."""
        rng = np.random.default_rng(42)
        fc = rng.uniform(-1.0, 1.0, N_STEPS)
        res = run_semi_cmd(P, fc, ROAD, t_end=1.0)
        fd_e, mask_e, v_rel = expected_channels(res, fc)
        n_out = int(res.mask.sum())
        print(f"[E1] T15 moderate: out-of-band = {n_out}/{N_STEPS} "
              f"({100.0 * n_out / N_STEPS:.1f}% command violation rate), "
              f"min F_d*v_rel = {np.min(res.fd * v_rel):.3e}")
        assert np.min(res.fd * v_rel) >= -1e-12
        assert np.array_equal(res.fd, fd_e)
        assert np.array_equal(res.mask, mask_e)
        assert 0 < n_out < N_STEPS   # mixed domain: both branches exercised

    @pytest.mark.parametrize("pattern", ["alternating", "const_pos", "const_neg"])
    def test_adversarial_commands(self, pattern):
        """Adversarial structures: sign flips every step / constant saturation."""
        if pattern == "alternating":
            fc = BIG * np.where(np.arange(N_STEPS) % 2 == 0, 1.0, -1.0)
        elif pattern == "const_pos":
            fc = np.full(N_STEPS, BIG)
        else:
            fc = np.full(N_STEPS, -BIG)
        res = run_semi_cmd(P, fc, ROAD, t_end=1.0)
        fd_e, mask_e, v_rel = expected_channels(res, fc)
        products = res.fd * v_rel
        print(f"[E1] T15 {pattern}: min F_d*v_rel = {np.min(products):.3e}, "
              f"out-of-band = {int(res.mask.sum())}/{N_STEPS}")
        assert np.min(products) >= -1e-12
        assert np.array_equal(res.fd, fd_e)
        assert np.array_equal(res.mask, mask_e)
        assert int(res.mask.sum()) == N_STEPS   # huge command: always clipped


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main(["-v", "-s", __file__]))
