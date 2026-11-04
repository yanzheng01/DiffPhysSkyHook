"""G4 gate tests: oracle convergence quality (plan D, T32).

docs/plan_d_training_optimization_dev_test_plan.md §3/G4:

  T32 artifact-based (docs/plan_d_g4_oracle.json):
    - artifact completeness: both modes x {pgd, planner}, 400-iteration
      j_sim trajectories on the held-out seed-777 fixture, final rows
      finite;
    - convergence criterion: the relative change of j_sim over each of
      the last two 50-iteration windows is < 1% (both oracles, both
      modes);
    - privileged oracle beats every causal controller: the converged
      planner's j_sim is strictly below the G3 adopted-policy anchors
      (hardcoded from docs/plan_d_g3_scale.json) on both modes -- the
      info-gap yardstick must stay positive after re-converging the
      oracle;
    - privileged/causal not mixed: every oracle entry carries
      privileged=True, the policy anchor privileged=False, and the
      yardstick is stated against the planner only.
"""

import json
import math
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARTIFACT = os.path.join(ROOT, "docs", "plan_d_g4_oracle.json")

# G3 adopted configuration (decay=1.0) held-out anchors, hardcoded so a
# silent edit of docs/plan_d_g3_scale.json cannot weaken this gate.
POLICY_ANCHOR = {"m1": 4.3299, "m2": 4.0729}


def _load():
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


def _runs():
    art = _load()
    for mode in (1, 2):
        for kind in ("pgd", "planner"):
            yield art[f"m{mode}_{kind}"], mode, kind


class TestT32OracleConvergence:
    def test_artifact_complete(self):
        art = _load()
        assert art["meta"]["eval_seed"] == 777 and art["meta"]["B"] == 8
        for run, mode, kind in _runs():
            assert run["privileged"] is True
            assert run["mode"] == mode and run["kind"] == kind
            assert run["n_iter"] == 400
            traj = run["traj"]
            iters = [p["iter"] for p in traj]
            assert iters[0] == 1 and iters[-1] == 400
            assert all(i2 - i1 == 50 for i1, i2 in
                       zip(iters[:-1], iters[1:-1]))
            for p in traj:
                assert math.isfinite(p["obj"])
                assert math.isfinite(p["j_sim"])
            for key in ("j_sim", "jerk", "J_acc", "ms_ft", "mask_rate"):
                assert math.isfinite(run[key])

    def test_convergence_criterion(self):
        """Last two 50-iter windows each improve j_sim by < 1%."""
        for run, mode, kind in _runs():
            j = [p["j_sim"] for p in run["traj"]]
            for k in (-2, -1):
                it_hi = run["traj"][k]["iter"]
                drop = (j[k - 1] - j[k]) / abs(j[k])
                print(f"[T32] m{mode} {kind}: window ending it{it_hi} "
                      f"drop {drop * 100:.4f}%")
                assert drop < 0.01, \
                    f"m{mode} {kind}: 50-iter window change " \
                    f"{drop * 100:.3f}% >= 1% (not converged)"

    def test_privileged_beats_all_causal(self):
        art = _load()
        for mode in ("m1", "m2"):
            planner = art[f"{mode}_planner"]["j_sim"]
            anchor = POLICY_ANCHOR[mode]
            gap = anchor - planner
            print(f"[T32] {mode}: planner {planner:.4f} vs policy "
                  f"{anchor:.4f} -> info gap {gap:+.4f}")
            assert planner < anchor, \
                f"{mode}: converged planner {planner:.4f} does not beat " \
                f"the causal policy anchor {anchor:.4f}"

    def test_privileged_causal_not_mixed(self):
        art = _load()
        for run, _mode, _kind in _runs():
            assert run["privileged"] is True
        assert art["policy_anchor"]["privileged"] is False
        # the yardstick is stated against the planner only (PGD is a
        # privileged upper bound without the training regularizers)
        assert set(art["policy_anchor"].keys()) == {
            "privileged", "source", "m1_j_sim", "m2_j_sim"}
