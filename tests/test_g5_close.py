"""G5 gate tests: close-out (plan D, T33).

docs/plan_d_training_optimization_dev_test_plan.md §3/G5:

  T33 artifact-based (docs/plan_d_g5_final.json):
    - final evaluation table completeness: both modes, the four causal
      rows (passive / continuous skyhook / ON-OFF skyhook / adopted G3
      policy) with full metric fields, privileged oracle rows carried
      over from G4 flagged privileged and kept OUT of the causal
      comparisons (T25 protocol: '*' discipline);
    - T25/T28-style assertions on the FINAL models: the adopted policy
      beats every causal baseline on j_sim on BOTH modes, and the
      comfort-trigger block is computed consistently from its own data
      (fires == all-modes-improvement < 2%).
"""

import json
import math
import os

CAUSAL = ("passive c_min", "continuous skyhook", "ON-OFF skyhook")
METRIC_FIELDS = ("j_sim", "J_acc", "jerk", "ms_ft", "mask_rate")


def _load():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "docs",
                        "plan_d_g5_final.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class TestT33CloseOut:
    def test_final_table_complete_and_privileged_flagged(self):
        art = _load()
        assert art["meta"]["eval_seed"] == 777 and art["meta"]["B"] == 8
        for mode in ("m1", "m2"):
            rows = art[mode]["rows"]
            names = [r["name"] for r in rows]
            for base in CAUSAL:
                assert base in names
            assert any(r["name"].startswith("policy") for r in rows)
            assert sum(r["privileged"] for r in rows) == 2, \
                f"{mode}: expected exactly 2 privileged rows (pgd+planner)"
            for r in rows:
                for k in METRIC_FIELDS:
                    assert math.isfinite(r[k])

    def test_policy_beats_causal_and_trigger_consistent(self):
        art = _load()
        for mode in ("m1", "m2"):
            rows = art[mode]["rows"]
            pol = next(r for r in rows if r["name"].startswith("policy"))
            causal = [r for r in rows if r["name"] in CAUSAL]
            for c in causal:
                assert pol["j_sim"] < c["j_sim"], \
                    f"{mode}: policy {pol['j_sim']:.4f} does not beat " \
                    f"{c['name']} {c['j_sim']:.4f}"
            assert not pol["privileged"]
        trig = art["comfort_trigger"]
        expect = all(v["improvement"] < 0.02 for v in trig["data"].values())
        assert trig["fires"] == expect, \
            "comfort-trigger verdict inconsistent with its own data"
