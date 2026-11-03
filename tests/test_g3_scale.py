"""G3 gate tests: training-dynamics scale-up (plan D, T31).

docs/plan_d_training_optimization_dev_test_plan.md §3/G3:

  T31 batch equivalence (single update, seed-aligned)
    - one B=32 episode loss + policy gradient equals the average of the
      four seed-aligned B=8 sub-batch losses / gradients (rollout batch
      invariance of the kernel + mean-reduction invariance of both
      J_sim and J_ctl). Four sequential SGD steps are NOT equivalent to
      one B=32 step -- the claim tested is the per-iteration one.
  T31 scale-up runs (artifact-based)
    - docs/plan_d_g3_scale.json exists with the four long runs
      (mode {1,2} x decay {1.0, 0.9}, B=32, 800 iters, GPU);
    - each entry: finite best select j_sim, best_iter within cadence,
      wall-clock recorded;
    - B=32 gradient-noise reduction: windowed std of the training
      j_sim tail (last 100 iters) is below the B=8 G2-anchor run's
      (variance of a batch mean over 32 is expected ~ 1/4 of B=8);
    - decay two-point comparison is recorded (no pass/fail -- an
      ablation, decision at G3 close-out).
"""

import json
import math
import os

import numpy as np
import pytest
import torch

from skyhook.params import SemiActiveParams
from skyhook.policy import PolicyMLP
from train_policy import TrainConfig, run_episode, sample_ics, sample_roads

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARTIFACT = os.path.join(ROOT, "docs", "plan_d_g3_scale.json")
ANCHOR_CK = os.path.join(ROOT, "checkpoints", "g2", "wj0.0001")


class TestT31BatchEquivalence:
    """One B=32 update == average of four seed-aligned B=8 updates."""

    B32, B8, H, T_W, DT = 32, 8, 50, 0.5, 1.0e-3   # K=10, fast on CPU

    def _fixture(self):
        cfg32 = TrainConfig(mode=1, B=self.B32, H=self.H, t_w=self.T_W,
                            seed=0, fixed_road_seed=None)
        rng = np.random.default_rng(0)
        roads32 = sample_roads(cfg32, rng)
        ics32 = sample_ics(cfg32, rng)
        cfg8 = TrainConfig(mode=1, B=self.B8, H=self.H, t_w=self.T_W,
                           seed=0, fixed_road_seed=None)
        return cfg32, cfg8, roads32, ics32

    def test_loss_and_gradient_equivalence(self):
        cfg32, cfg8, roads32, ics32 = self._fixture()
        params32 = [SemiActiveParams() for _ in range(self.B32)]
        xmaxs32 = [0.08] * self.B32

        torch.manual_seed(0)
        pol32 = PolicyMLP(self.H)
        loss32, jctl32, _ = run_episode(pol32, cfg32, roads32, ics32,
                                        params32, xmaxs32)
        (loss32 + jctl32).backward()
        g32 = {n: p.grad.detach().clone()
               for n, p in pol32.named_parameters()}

        torch.manual_seed(0)
        pol8 = PolicyMLP(self.H)
        total = 0.0
        for b in range(4):
            sl = slice(b * self.B8, (b + 1) * self.B8)
            loss_b, jctl_b, _ = run_episode(
                pol8, cfg8, roads32[sl], ics32[sl],
                params32[sl], xmaxs32[sl])
            ((loss_b + jctl_b) / 4.0).backward()
            total += float(loss_b.item() + jctl_b.item()) / 4.0
        g8 = {n: p.grad.detach().clone()
              for n, p in pol8.named_parameters()}

        l32 = float(loss32.item() + jctl32.item())
        print(f"[T31] B=32 loss {l32:.9e} vs 4xB=8 mean {total:.9e}")
        assert abs(l32 - total) <= 1e-9 * max(1.0, abs(l32))
        for n in g32:
            d = torch.max(torch.abs(g32[n] - g8[n])).item()
            s = torch.max(torch.abs(g32[n])).item()
            print(f"[T31] grad[{n}]: max|diff|={d:.3e} scale={s:.3e}")
            assert d <= 1e-9 * max(1.0, s)


def _load():
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


class TestT31ScaleUpArtifact:
    def test_artifact_complete(self):
        scan = _load()
        for mode in (1, 2):
            for decay in ("1.0", "0.9"):
                e = scan[f"m{mode}_decay{decay}"]
                assert e["B"] == 32 and e["n_iter"] == 800
                # smoke calibration: CUDA 3.95 s/iter vs CPU 2.19 s/iter
                # on the GTX 1060 (kernel-launch bound) -> CPU route
                assert e["device"] == "cpu"
                assert math.isfinite(e["best_select_j_sim"])
                assert 0 < e["best_iter"] <= 800
                assert e["best_iter"] % 25 == 0 or e["best_iter"] == 800
                assert e["wall_s"] > 0.0

    def test_gradient_noise_reduction_vs_b8(self):
        scan = _load()
        for mode in (1, 2):
            g3 = _hist_std(scan[f"m{mode}_decay1.0"]["train_j_sim_tail"])
            anchor = _hist_std(
                _anchor_hist(mode, "train_j_sim_tail"))
            print(f"[T31] m{mode}: B=32 tail std {g3:.4f} vs "
                  f"B=8 anchor {anchor:.4f}")
            assert g3 < anchor, \
                f"m{mode}: B=32 tail std {g3:.4f} not below B=8 {anchor:.4f}"

    def test_decay_ablation_recorded(self):
        scan = _load()
        for mode in (1, 2):
            a = scan[f"m{mode}_decay1.0"]
            b = scan[f"m{mode}_decay0.9"]
            assert math.isfinite(a["best_select_j_sim"])
            assert math.isfinite(b["best_select_j_sim"])
            assert a["wall_s"] > 0.0 and b["wall_s"] > 0.0


def _hist_std(tail):
    return float(np.std(tail))


def _anchor_hist(mode, key):
    """B=8 (G2 adopted arm) training-j_sim tail from its checkpoint."""
    ck = torch.load(os.path.join(ANCHOR_CK, f"policy_m{mode}_best.pt"),
                    weights_only=False)
    hist = ck["history"]["j_sim"]
    it = ck["best_iter"]
    tail = hist[max(0, it - 100):it]
    assert len(tail) >= 50
    return tail
