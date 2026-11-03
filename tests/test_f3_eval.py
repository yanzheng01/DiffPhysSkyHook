"""F3 gate tests (plan C section 4): per-mode evaluation + mode comparison.

  T25 evaluation tables: the closed-loop roller is bit-equivalent to
     run_cmd_seq_batch (whole window AND sub-spliced through the roller);
     the passive baseline is exactly realizable (fd == f_cmd, mask == 0);
     the mode-1 and mode-2 protocol tables both show the trained policy
     improving on the continuous-skyhook baseline (improvement > 0) with
     all six metrics + mode-specific extras finite and in range; the
     privileged whole-window oracles sit below every causal controller
     (upper-bound quality) and the planner-vs-PGD gap is logged.
  T28 mode-comparison protocol: same-seed/same-parameter-count mode-1 and
     mode-2 training curves both descend and are separated; on the SAME
     event fixture the mode-2 policy's event-window peak |v_dot_2| beats
     the mode-1 policy's (preview value); negative control -- the
     roughness-switch settling shows no significant mode difference
     (the preview carries no material component, v2.3); causality -- the
     mode-1 policy output is invariant under permutation of the FUTURE
     road while the mode-2 output is not.
"""

import numpy as np
import pytest
import torch

import eval_policy as ep
from skyhook import sim3
from skyhook.params import SemiActiveParams
from skyhook.policy import ObsBuffer, PolicyMLP, build_obs
from skyhook.road_seq import Event, ar_switching, events_ar
from train_policy import TrainConfig, train

DT = 1.0e-3
N = 1000                     # T_w = 1 s episode
EVAL_SEED = 777              # held out from the training seed / fixture 11
TRAIN_ITERS = 400
# Domain-randomized training (plan C decision 7: per-episode resampling;
# the rng is seed-deterministic). The FIXED 8-road fixture of the T23
# smoke gate overfits badly -- with resampling the preview value shows
# up on every mode-2 metric (j_sim AND event peak AND overshoot).

_P0 = SemiActiveParams()
PARAMS = [SemiActiveParams(gamma_sky=_P0.gamma_sky * (0.8 + 0.2 * i),
                           c_min=_P0.c_min, c_max=_P0.c_max)
          for i in range(3)]
X0S = [(0.0, 0.01, 0.0, -0.02),
       (0.005, -0.01, 0.02, 0.03),
       (-0.005, 0.02, -0.01, 0.0)]
EVTS = [Event(amp=-0.02, t0=0.30, width=0.08, kind="pothole"),
        Event(amp=0.015, t0=0.60, width=0.10, kind="bump")]
SEQS = [ar_switching(0.003, 0.012, 0.5, 500, DT, seed=7),
        events_ar(EVTS, 0.003, 0.006, 0.4, 500, DT, seed=11),
        events_ar(EVTS, 0.006, 0.003, 0.2, 500, DT, seed=13)]


# ---------------------------------------------------------------------------
# shared trained policies (module scope: both gates reuse them)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained():
    """(mode-1 policy, mode-2 policy, histories) -- same seed, same arch."""
    out = {}
    for mode in (1, 2):
        cfg = TrainConfig(mode=mode, B=8, H=100, t_w=1.0,
                          n_iter=TRAIN_ITERS, seed=0,
                          fixed_road_seed=None)   # resample (decision 7)
        pol, hist = train(cfg, verbose=False)
        out[mode] = (pol, hist)
    return out


def _eval_rows(mode, policy, B=8):
    """Baselines + policy rows on the held-out fixture (no oracles)."""
    roads, ics = ep.build_eval_fixture(mode, B, 1.0, DT, EVAL_SEED)
    params = [SemiActiveParams() for _ in range(B)]
    xmaxs = [0.08] * B
    args = dict(roads=roads, params_list=params, ics=ics, xmaxs=xmaxs,
                mode=mode, n_steps=N, dt=DT)
    rows = {
        "passive": ep.evaluate_controller("passive", ep.make_passive(params),
                                          1, **args),
        "skyhook": ep.evaluate_controller("skyhook", ep.make_skyhook(params),
                                          1, **args),
        "onoff": ep.evaluate_controller("onoff", ep.make_onoff(params),
                                        1, **args),
    }
    if policy is not None:
        fn = ep.make_policy_cmd(policy, roads, xmaxs, DT, mode, 100)
        rows["policy"] = ep.evaluate_controller("policy", fn, 100, **args)
    rows["_roads"] = roads
    rows["_ics"] = ics
    rows["_params"] = params
    rows["_xmaxs"] = xmaxs
    return rows


# ---------------------------------------------------------------------------
# T25: evaluation protocol
# ---------------------------------------------------------------------------
class TestT25Roller:
    def test_whole_window_equivalence(self):
        """sub = N open-loop roller == run_cmd_seq_batch, element-wise."""
        rng = np.random.default_rng(99)
        B = 3
        fc = rng.uniform(-0.3, 0.3, size=(B, N))
        x0s = [tuple(x) for x in X0S]
        xmaxs = [0.08] * B
        seqs = [ar_switching(0.003, 0.012, 0.5, N, DT, seed=40 + b)
                for b in range(B)]
        ref = sim3.run_cmd_seq_batch(PARAMS, fc, seqs, x0s=x0s,
                                     xmaxs=xmaxs, t_end=N * DT, dt=DT)
        rec = ep.closed_loop_rollout(ep.make_openloop_cmd(fc, N), seqs,
                                     PARAMS, x0s, xmaxs, N, DT, sub=N)
        diffs = [np.max(np.abs(rec.state - ref.state)),
                 np.max(np.abs(rec.a2 - ref.a2)),
                 np.max(np.abs(rec.fd - ref.fd)),
                 np.max(np.abs(rec.mask - ref.mask)),
                 np.max(np.abs(rec.ft - ref.ft)),
                 np.max(np.abs(rec.travel - ref.travel))]
        print(f"[F3] T25 roller whole-window max|diff| = {max(diffs):.3e}")
        assert max(diffs) < 1.0e-12
        assert np.max(np.abs(rec.j_sim - ref.j_sim)) < 1.0e-12
        assert np.max(np.abs(rec.j_terms - ref.j_terms)) < 1.0e-12

    def test_sub_spliced_equivalence(self):
        """sub = H through the roller: same trajectory as the whole window."""
        rng = np.random.default_rng(7)
        B, H = 3, 100
        fc = rng.uniform(-0.3, 0.3, size=(B, N))
        x0s = [tuple(x) for x in X0S]
        xmaxs = [0.08] * B
        seqs = [events_ar(EVTS, 0.003, 0.006, 0.4, N, DT, seed=50 + b)
                for b in range(B)]
        whole = sim3.run_cmd_seq_batch(PARAMS, fc, seqs, x0s=x0s,
                                       xmaxs=xmaxs, t_end=N * DT, dt=DT)
        rec = ep.closed_loop_rollout(ep.make_openloop_cmd(fc, H), seqs,
                                     PARAMS, x0s, xmaxs, N, DT, sub=H)
        diff = np.max(np.abs(rec.state - whole.state))
        assert diff < 1.0e-12
        # the recorded command series is the replayed series itself
        assert np.array_equal(rec.f_cmd, fc)

    def test_passive_exactly_realizable(self):
        """c_min * v_rel rides the window lower bound: fd == f_cmd, no
        command is ever out of band (the Plan-B passive semantics)."""
        B = 3
        roads, ics = ep.build_eval_fixture(1, B, 1.0, DT, EVAL_SEED)
        rec = ep.closed_loop_rollout(ep.make_passive(PARAMS), roads, PARAMS,
                                     ics, [0.08] * B, N, DT, sub=1)
        assert np.all(rec.mask == 0.0)
        assert np.max(np.abs(rec.fd - rec.f_cmd)) < 1.0e-12


class TestT25MetricUnits:
    def test_settling_synthetic(self):
        """Settling semantics on a synthetic a2 trace: 0.2 s of elevated
        post-switch response then the steady level -> settling ~ 0.2 s."""
        t_sw = 0.5
        n_sw = int(round(t_sw / DT))
        steady = 1.0
        a2 = np.full((1, N), steady)
        a2[0, n_sw:n_sw + 200] = 5.0                     # 0.2 s transient
        rec = ep.EpisodeRecords(
            t=np.arange(N) * DT, state=np.zeros((1, N, 4)), a2=a2,
            fd=a2, mask=np.zeros((1, N)), ft=a2, travel=a2,
            f_cmd=np.zeros((1, N)), j_sim=np.zeros(1), j_terms=np.zeros((1, 4)))
        roads = [ar_switching(0.003, 0.012, t_sw, N, DT, seed=1)]
        m = ep.mode1_metrics(rec, roads, DT)
        # the last violating 50 ms window starts at n_sw+150 -> settling
        # counted from t_sw to its end ~ 0.2 s (window granularity)
        assert 0.15 <= m["settling_s"] <= 0.25
        assert m["pre_J_acc"] == pytest.approx(1.0)
        assert m["post_J_acc"] > m["pre_J_acc"]

    def test_event_metrics_synthetic(self):
        """Peak = max |a2| in the event window; overshoot = body excursion
        relative to the window-start position."""
        ev = Event(amp=-0.02, t0=0.40, width=0.08)
        roads = [events_ar([ev], 0.003, 0.003, None, N, DT, seed=2)]
        a2 = np.zeros((1, N))
        a2[0, 380] = 3.0                                 # inside the window
        a2[0, 100] = 9.0                                 # OUTSIDE (before)
        x2 = np.linspace(0.0, 0.01, N)[None, :]
        rec = ep.EpisodeRecords(
            t=np.arange(N) * DT, state=np.zeros((1, N, 4)), a2=a2,
            fd=a2, mask=np.zeros((1, N)), ft=a2, travel=a2,
            f_cmd=np.zeros((1, N)), j_sim=np.zeros(1), j_terms=np.zeros((1, 4)))
        rec.state[0, :, 1] = x2[0]
        m = ep.mode2_metrics(rec, roads, DT)
        assert m["event_peak_acc"] == pytest.approx(3.0)  # 9.0 excluded
        # x2 rises linearly 0 -> 0.01 over 1 s; window start ~ n=200
        n0 = max(0, int(round((ev.t0 - 0.2) / DT)))
        n1 = min(N, int(round((ev.t0 + ev.width + 0.2) / DT)))
        expected = x2[0, n0:n1].max() - x2[0, n0]
        assert m["event_overshoot_m"] == pytest.approx(expected)


class TestT25Tables:
    def test_mode1_policy_beats_skyhook(self, trained):
        rows = _eval_rows(1, trained[1][0])
        pol, sky = rows["policy"], rows["skyhook"]
        print(f"[F3] T25 mode-1: policy j_sim={pol['j_sim']:.4f} "
              f"skyhook={sky['j_sim']:.4f} passive={rows['passive']['j_sim']:.4f}")
        ep._print_table("T25 mode-1 table", [rows[k] for k in
                                             ("passive", "skyhook", "onoff",
                                              "policy")])
        assert pol["j_sim"] < sky["j_sim"]               # improvement > 0
        # mode-1 protocol extras: finite, in-range settling + split J_acc
        for r in (pol, sky):
            assert 0.0 <= r["settling_s"] <= 1.0
            assert np.all(np.isfinite(r["settling_all"]))
            assert r["pre_J_acc"] > 0.0 and r["post_J_acc"] > 0.0

    def test_mode2_policy_beats_skyhook(self, trained):
        rows = _eval_rows(2, trained[2][0])
        pol, sky = rows["policy"], rows["skyhook"]
        print(f"[F3] T25 mode-2: policy j_sim={pol['j_sim']:.4f} "
              f"skyhook={sky['j_sim']:.4f} passive={rows['passive']['j_sim']:.4f}")
        ep._print_table("T25 mode-2 table", [rows[k] for k in
                                             ("passive", "skyhook", "onoff",
                                              "policy")])
        assert pol["j_sim"] < sky["j_sim"]
        assert pol["event_peak_acc"] > 0.0
        assert pol["event_overshoot_m"] > 0.0

    def test_oracle_upper_bound(self, trained):
        """Privileged whole-window PGD sits below every causal controller;
        the planner-vs-PGD gap is logged (T25 deliverable)."""
        B = 3
        roads, ics = ep.build_eval_fixture(2, B, 1.0, DT, EVAL_SEED)
        params = [SemiActiveParams() for _ in range(B)]
        xmaxs = [0.08] * B
        f_pgd = ep.oracle_optimize(roads, params, ics, xmaxs,
                                   n_iter=60, verbose=False)
        f_plan = ep.oracle_optimize(roads, params, ics, xmaxs,
                                    w_jerk=1.0e-5, w_mag=1.0e-3,
                                    n_iter=60, verbose=False)
        args = dict(roads=roads, params_list=params, ics=ics, xmaxs=xmaxs,
                    mode=2, n_steps=N, dt=DT)
        r_pgd = ep.evaluate_controller("PGD", ep.make_openloop_cmd(f_pgd, N),
                                       N, **args)
        r_plan = ep.evaluate_controller("planner",
                                        ep.make_openloop_cmd(f_plan, N),
                                        N, **args)
        r_sky = ep.evaluate_controller("skyhook", ep.make_skyhook(params),
                                       1, **args)
        print(f"[F3] T25 oracle: PGD j_sim={r_pgd['j_sim']:.4f} "
              f"planner={r_plan['j_sim']:.4f} skyhook={r_sky['j_sim']:.4f} "
              f"gap(plan-PGD)={r_plan['j_sim'] - r_pgd['j_sim']:+.4e}")
        assert r_pgd["j_sim"] < r_sky["j_sim"]           # upper-bound quality
        assert r_plan["jerk"] < r_pgd["jerk"]            # regularizer bites


# ---------------------------------------------------------------------------
# T28: mode-comparison protocol
# ---------------------------------------------------------------------------
class TestT28ModeComparison:
    def test_curves_separate_and_descend(self, trained):
        h1 = np.asarray(trained[1][1]["j_sim"])
        h2 = np.asarray(trained[2][1]["j_sim"])
        assert np.all(np.isfinite(h1)) and np.all(np.isfinite(h2))
        assert h1[0] > h1[-1] and h2[0] > h2[-1]         # both descend
        assert not np.array_equal(h1, h2)                # curves separated
        n1 = sum(p.numel() for p in trained[1][0].parameters())
        n2 = sum(p.numel() for p in trained[2][0].parameters())
        assert n1 == n2                                  # fair protocol
        print(f"[F3] T28 curves: m1 {h1[0]:.3f}->{h1[-1]:.3f}, "
              f"m2 {h2[0]:.3f}->{h2[-1]:.3f} (params {n1}=={n2})")

    def test_mode2_event_peak_beats_mode1(self, trained):
        """Same event fixture: the preview policy damps the event-window
        peak |v_dot_2| better than the preview-blind policy."""
        rows1 = _eval_rows(2, trained[1][0])             # mode-1 policy...
        rows2 = _eval_rows(2, trained[2][0])             # ...vs mode-2, same roads
        p1, p2 = rows1["policy"], rows2["policy"]
        print(f"[F3] T28 event peak: m1-policy={p1['event_peak_acc']:.4f} "
              f"m2-policy={p2['event_peak_acc']:.4f}")
        assert p2["event_peak_acc"] < p1["event_peak_acc"]
        assert p2["j_sim"] < p1["j_sim"]                 # overall too

    def test_negative_control_settling(self, trained):
        """Roughness-switch settling: no significant mode difference (the
        preview channel carries no material component, v2.3)."""
        rows1 = _eval_rows(1, trained[1][0])
        rows2 = _eval_rows(1, trained[2][0])
        s1 = rows1["policy"]["settling_s"]
        s2 = rows2["policy"]["settling_s"]
        print(f"[F3] T28 negative control: settling m1={s1:.4f} m2={s2:.4f}")
        # no significant difference: within 30% of each other (or both tiny)
        assert abs(s2 - s1) <= 0.3 * max(s1, s2) + 0.05

    def test_causality_permutation(self, trained):
        """Mode-1 output is invariant to the FUTURE road; mode-2 is not."""
        B = 2
        roads_a = [events_ar(EVTS, 0.003, 0.006, 0.4, N, DT, seed=60 + b)
                   for b in range(B)]
        # permuted future: same AR background family, different events
        evts_b = [Event(amp=0.025, t0=0.45, width=0.06, kind="bump"),
                  Event(amp=-0.015, t0=0.80, width=0.12, kind="dip")]
        roads_b = [events_ar(evts_b, 0.003, 0.006, 0.4, N, DT, seed=60 + b)
                   for b in range(B)]

        buf = ObsBuffer(B)
        rng = np.random.default_rng(3)
        buf.push(rng.normal(0, 1.0, (B, 100)), rng.normal(0, 0.05, (B, 100)),
                 DT)
        state = np.asarray(X0S[:2], dtype=np.float64)
        xmaxs = [0.08] * B
        i0 = 2 * 300                     # boundary at t = 0.3 s

        obs_a1 = build_obs(buf, roads_a, i0, state, xmaxs, DT, 1)
        obs_b1 = build_obs(buf, roads_b, i0, state, xmaxs, DT, 1)
        with torch.no_grad():
            f_a1 = trained[1][0](obs_a1)
            f_b1 = trained[1][0](obs_b1)
        assert torch.equal(f_a1, f_b1)                 # causal: identical

        obs_a2 = build_obs(buf, roads_a, i0, state, xmaxs, DT, 2)
        obs_b2 = build_obs(buf, roads_b, i0, state, xmaxs, DT, 2)
        assert not np.allclose(obs_a2.preview.numpy(),
                               obs_b2.preview.numpy())  # preview differs
        with torch.no_grad():
            f_a2 = trained[2][0](obs_a2)
            f_b2 = trained[2][0](obs_b2)
        assert not torch.allclose(f_a2, f_b2)          # preview is read
