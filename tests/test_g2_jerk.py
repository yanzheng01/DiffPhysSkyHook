"""G2 gate tests: w_jerk magnitude scan (plan D, T30).

docs/plan_d_training_optimization_dev_test_plan.md §3/G2 (as amended in
§8: the J_ctl regularizer exists since plan C / T24; G2 is a w_jerk
magnitude scan {1e-6, 1e-5, 1e-4} on both modes, anchored at the G1
best checkpoints which ARE the w_jerk = 1e-5 arm).

  T30 artifact integrity
    - docs/plan_d_g2_wjerk_scan.json exists and covers 3 arms x 2 modes;
    - every entry carries finite j_sim / jerk / best_iter and the
      arm's w_jerk;
    - the w_jerk = 1e-5 arm matches the G1 best checkpoints (m1 it400,
      m2 it350) -- the anchor identity.
  T30 acceptance (the G2 gate proper)
    - arm selection rule: among arms within the 5% j_sim budget relative
      to the 1e-5 anchor, pick the lowest eval jerk;
    - the selected arm must cut jerk by >= 10x vs the anchor AND stay
      within the j_sim budget, for BOTH modes;
    - stronger regularization must not increase jerk: jerk(1e-4 arm)
      <= jerk(1e-6 arm) per mode.
  T30 gradient/boundary semantics
    - covered by plan C's T24 (tests/test_f2_policy.py: autograd vs FD,
      spliced boundary jump penalized exactly like an in-segment jump,
      boundary coupling of consecutive blocks). Not duplicated here.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARTIFACT = os.path.join(ROOT, "docs", "plan_d_g2_wjerk_scan.json")

ARMS = ("1e-06", "1e-05", "1e-04", "3e-04", "1e-03")
MODES = (1, 2)
JSIM_BUDGET = 1.05          # 5% degradation budget vs the 1e-5 anchor
JERK_FACTOR = 2.0           # amended gate (plan D §8 G2, decision A):
                            # 10x proven infeasible within budget across
                            # 5 arms / 3 decades; adopted arm = 1e-04
ADOPTED = "1e-04"           # Pareto: j_sim improves on BOTH modes


def _load():
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


def _entry(scan, mode, arm):
    return scan[f"m{mode}_{arm}"]


class TestT30Artifact:
    def test_scan_artifact_complete(self):
        scan = _load()
        for mode in MODES:
            for arm in ARMS:
                e = _entry(scan, mode, arm)
                assert math.isfinite(e["j_sim"]) and e["j_sim"] > 0.0
                assert math.isfinite(e["jerk"]) and e["jerk"] > 0.0
                assert isinstance(e["best_iter"], int) and e["best_iter"] > 0
                assert float(e["w_jerk"]) == float(arm)

    def test_anchor_is_g1_best(self):
        scan = _load()
        anchor = {1: 400, 2: 350}          # G1 best_iter (§8 table)
        for mode in MODES:
            e = _entry(scan, mode, "1e-05")
            assert e["best_iter"] == anchor[mode]
            assert e["source"] == "checkpoints"   # G1 best path


class TestT30Acceptance:
    def test_adopted_arm_meets_gate_both_modes(self):
        """Amendment A: the pinned adopted arm (w_jerk = 1e-4) must cut
        jerk by >= 2x vs the 1e-5 anchor and stay within the j_sim
        budget, on BOTH modes (single shared w_jerk, fair-comparison
        protocol)."""
        scan = _load()
        for mode in MODES:
            a = _entry(scan, mode, "1e-05")
            e = _entry(scan, mode, ADOPTED)
            assert e["jerk"] * JERK_FACTOR <= a["jerk"], \
                f"m{mode}: adopted arm jerk {e['jerk']:.3e} not " \
                f"{JERK_FACTOR:g}x below anchor {a['jerk']:.3e}"
            assert e["j_sim"] <= JSIM_BUDGET * a["j_sim"], mode

    def test_infeasibility_of_10x_documented(self):
        """Negative control for the amendment: no arm within the j_sim
        budget achieves the original 10x jerk reduction (the evidence
        that justified amending the gate)."""
        scan = _load()
        for mode in MODES:
            a = _entry(scan, mode, "1e-05")
            for arm in ARMS:
                e = _entry(scan, mode, arm)
                if e["j_sim"] <= JSIM_BUDGET * a["j_sim"]:
                    assert e["jerk"] * 10.0 > a["jerk"], (mode, arm)

    def test_stronger_regularization_reduces_jerk(self):
        scan = _load()
        for mode in MODES:
            assert (_entry(scan, mode, "1e-04")["jerk"]
                    <= _entry(scan, mode, "1e-06")["jerk"]), mode
