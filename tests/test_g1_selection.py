"""G1 gate tests: frozen-batch select evaluation + best checkpoint
(plan D, T29 -- docs/plan_d_training_optimization_dev_test_plan.md §3/G1).

  T29 selection determinism
    - two trainings with identical cfg produce bit-identical histories
      (train j_sim AND select curves) and the same best_iter;
    - the select evaluation is side-effect free on the training stream:
      varying select_every does not change the training loss sequence.
  T29 select-seed isolation (555 never enters the training chain)
    - the select fixture is deterministic across constructions;
    - the select batch differs from the seed-0 training batch;
    - eval_select runs under no_grad and returns a plain float (no
      graph, no backward, policy .grad untouched).
  T29 best-checkpoint structure
    - keys {model, cfg, history, best_iter, select_j_sim};
    - loads into PolicyMLP; best_iter on the select cadence;
    - select_j_sim == min(select curve); curve length == cadence count.
  T29 eval-seed (777) untouched -- static assertion
    - the training source never references the test seed 777; the only
      sanctioned select seed constant is 555.
"""

import os

import numpy as np
import pytest
import torch

from skyhook.policy import PolicyMLP
from train_policy import (SELECT_SEED, TrainConfig, eval_select, sample_ics,
                          sample_roads, train)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN_SRC = os.path.join(ROOT, "train_policy.py")


def _tiny_cfg(tmp_path, **kw):
    cfg = TrainConfig(mode=1, B=2, H=25, t_w=0.25, n_iter=6,
                      select_every=2, ckpt_every=1000, seed=0,
                      fixed_road_seed=None,
                      ckpt_dir=str(tmp_path / "ckpt") if tmp_path else "ckpt")
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class TestT29Determinism:
    def test_two_runs_identical(self, tmp_path):
        _, h1 = train(_tiny_cfg(tmp_path / "a"), verbose=False)
        _, h2 = train(_tiny_cfg(tmp_path / "b"), verbose=False)
        for key in ("j_sim", "select_j_sim", "jerk", "mask_rate"):
            assert h1[key] == h2[key], key
        c1 = torch.load(tmp_path / "a" / "ckpt" / "policy_m1_best.pt",
                        weights_only=False)
        c2 = torch.load(tmp_path / "b" / "ckpt" / "policy_m1_best.pt",
                        weights_only=False)
        assert c1["best_iter"] == c2["best_iter"]
        assert c1["select_j_sim"] == c2["select_j_sim"]

    def test_select_cadence_does_not_touch_training_stream(self, tmp_path):
        """Same seed, different select_every -> identical train losses."""
        _, h_ref = train(_tiny_cfg(tmp_path / "ref", select_every=10_000),
                         verbose=False)          # select only at final it
        _, h_sel = train(_tiny_cfg(tmp_path / "sel", select_every=2),
                         verbose=False)          # select at 2/4/6
        assert h_ref["j_sim"] == h_sel["j_sim"]
        assert h_ref["jerk"] == h_sel["jerk"]


class TestT29SeedIsolation:
    def test_select_fixture_frozen_and_distinct_from_train(self):
        cfg = _tiny_cfg(None)
        rng_a = np.random.default_rng(SELECT_SEED)
        roads_a = sample_roads(cfg, rng_a)
        ics_a = sample_ics(cfg, rng_a)
        rng_b = np.random.default_rng(SELECT_SEED)
        roads_b = sample_roads(cfg, rng_b)
        ics_b = sample_ics(cfg, rng_b)
        # frozen: two constructions identical
        for ra, rb in zip(roads_a, roads_b):
            assert np.array_equal(ra.z, rb.z) and np.array_equal(ra.zd, rb.zd)
        assert np.array_equal(ics_a, ics_b)
        # distinct from the seed-0 training stream's first episode
        rng_t = np.random.default_rng(0)
        roads_t = sample_roads(cfg, rng_t)
        ics_t = sample_ics(cfg, rng_t)
        assert not np.array_equal(ics_a, ics_t)
        assert any(not np.array_equal(ra.z, rt.z)
                   for ra, rt in zip(roads_a, roads_t))

    def test_eval_select_is_gradient_free(self, tmp_path):
        cfg = _tiny_cfg(tmp_path)
        policy = PolicyMLP(cfg.H)
        rng = np.random.default_rng(SELECT_SEED)
        roads = sample_roads(cfg, rng)
        ics = sample_ics(cfg, rng)
        from skyhook.params import SemiActiveParams
        params_list = [SemiActiveParams() for _ in range(cfg.B)]
        xmaxs = [0.08] * cfg.B
        out = eval_select(policy, cfg, roads, ics, params_list, xmaxs)
        assert isinstance(out, float)          # no tensor escapes no_grad
        for p in policy.parameters():
            assert p.grad is None              # no backward ever ran


class TestT29BestCheckpoint:
    def test_structure_and_consistency(self, tmp_path):
        cfg = _tiny_cfg(tmp_path)
        _, hist = train(cfg, verbose=False)
        path = os.path.join(cfg.ckpt_dir, "policy_m1_best.pt")
        assert os.path.exists(path)
        ck = torch.load(path, weights_only=False)
        assert {"model", "cfg", "history", "best_iter",
                "select_j_sim"} <= set(ck.keys())
        # cadence: select at 2/4/6 for n_iter=6, select_every=2
        assert len(hist["select_j_sim"]) == 3
        assert ck["best_iter"] in (2, 4, 6)
        assert ck["select_j_sim"] == pytest.approx(
            min(hist["select_j_sim"]))
        # model loads into a fresh PolicyMLP
        policy = PolicyMLP(cfg.H)
        policy.load_state_dict(ck["model"])
        assert ck["cfg"]["select_seed"] == SELECT_SEED


class TestT29EvalSeedUntouched:
    def test_train_source_never_references_777(self):
        src = open(TRAIN_SRC, encoding="utf-8").read()
        assert "777" not in src
        assert SELECT_SEED == 555
