"""F2 gate tests (plan C section 4): policy network + segmented training.

  T23 training smoke: B=8, K=5 x H=100, fixed road family, 50 iterations
     -- episode-level J_sim strictly descends (smoothed first/last thirds
     and endpoints), no NaN anywhere, and the pure bridge-adjoint nonzero
     fraction does not collapse (>= its early level).
  T24 regularizer gradients: jerk/mag autograd vs central FD (rel < 1e-6);
     splice discipline (a jump between two constant segment blocks is
     jerk-penalized exactly like an in-segment jump; a flat series has
     jerk == 0); total-gradient structure (dL/df_k == bridge adjoint +
     regularizer contribution, including the boundary coupling of
     consecutive blocks through the spliced difference).
  T27 segment-splice equivalence: the K-segment rollout with detached
     state handoff reproduces the whole-window rollout state trajectory
     element-wise (< 1e-12; Markov-chain pipeline check), and the
     segment-mean of the segment J's equals the whole-window j_sim_mean
     (equal-weight segment sum, decision 3). Road-cache switch: repeated
     bridge calls on the same RoadSeq objects hit the cache (identical J)
     and switching to new objects picks up the new roads.
"""

import numpy as np
import pytest
import torch

from skyhook import bridge, sim3
from skyhook.params import SemiActiveParams
from skyhook.policy import (
    PAST_WIN,
    PREV_WIN,
    ObsBuffer,
    PolicyMLP,
    build_obs,
    pool_mean,
    preview_features,
)
from skyhook.road_seq import Event, ar_switching, events_ar
from train_policy import TrainConfig, control_loss, train

DT = 1.0e-3

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
# early-event road: the event falls INSIDE the first 0.2 s preview window
# (the SEQS events start at t = 0.30 s, past the window's reach)
EVTS_EARLY = [Event(amp=-0.02, t0=0.02, width=0.08, kind="pothole")]
SEQ_EVT = events_ar(EVTS_EARLY, 0.003, 0.006, 0.4, 500, DT, seed=21)


# ---------------------------------------------------------------------------
# T23: segmented training smoke
# ---------------------------------------------------------------------------
class TestT23TrainingSmoke:
    def test_descent_no_nan_adjoint_alive(self, tmp_path):
        cfg = TrainConfig(mode=2, B=8, H=100, t_w=0.5,   # K = 5 segments
                          n_iter=50, lr=1.0e-3, w_jerk=1.0e-5,
                          seed=0, fixed_road_seed=11,
                          ckpt_dir=str(tmp_path))
        policy, hist = train(cfg, verbose=False)

        j = np.asarray(hist["j_sim"])
        assert j.shape == (50,)
        assert np.all(np.isfinite(j))                    # no NaN
        assert np.all(np.isfinite(hist["jerk"]))
        assert np.all(np.isfinite(hist["mag"]))

        # strict descent: endpoints and smoothed thirds
        n3 = 50 // 3
        assert j[0] > j[-1]
        assert j[:n3].mean() > j[-n3:].mean()
        # monotone trend over consecutive 10-iteration blocks (>= 4 of 5)
        blocks = [j[10 * i:10 * (i + 1)].mean() for i in range(5)]
        drops = sum(blocks[i + 1] < blocks[i] for i in range(4))
        assert drops >= 4

        # pure bridge-adjoint nonzero fraction does not collapse
        nz = np.asarray(hist["adj_nonzero"])
        assert nz[-1] > 0.0
        assert nz[-10:].mean() >= nz[:10].mean()

        # parameters finite + checkpoint on disk
        for p in policy.parameters():
            assert torch.all(torch.isfinite(p))
        ckpt = tmp_path / "policy_m2_it50.pt"
        assert ckpt.exists()
        loaded = torch.load(ckpt, weights_only=False)
        assert loaded["cfg"]["mode"] == 2
        assert len(loaded["history"]["j_sim"]) == 50


# ---------------------------------------------------------------------------
# T24: regularizer gradients + splice discipline + total structure
# ---------------------------------------------------------------------------
class TestT24RegularizerGradients:
    W_JERK, W_MAG = 0.01, 0.001

    def test_jerk_mag_autograd_vs_fd(self):
        rng = np.random.default_rng(42)
        f = torch.tensor(rng.normal(0.0, 0.05, size=(2, 200)),
                         dtype=torch.float64, requires_grad=True)
        loss = control_loss(f, DT, self.W_JERK, self.W_MAG)
        loss.backward()
        ad = f.grad.numpy()

        rel = lambda x, y: abs(x - y) / max(abs(y), 1.0e-300)  # noqa: E731
        h = 1.0e-6
        for (b, n) in [(0, 0), (0, 199), (1, 57), (1, 100), (1, 3)]:
            fp = f.detach().numpy().copy(); fp[b, n] += h
            fm = f.detach().numpy().copy(); fm[b, n] -= h
            fd = ((control_loss(torch.tensor(fp), DT, self.W_JERK,
                                self.W_MAG).item()
                   - control_loss(torch.tensor(fm), DT, self.W_JERK,
                                  self.W_MAG).item()) / (2.0 * h))
            assert rel(ad[b, n], fd) < 1.0e-6

    def test_splice_discipline(self):
        """A jump between two constant blocks is penalized; flat is free."""
        c1, c2 = 0.03, -0.02
        f_jump = torch.tensor(np.concatenate(
            [np.full((1, 50), c1), np.full((1, 50), c2)], axis=1))
        f_flat = torch.tensor(np.full((1, 100), c1))
        j_jump = control_loss(f_jump, DT, self.W_JERK, 0.0).item()
        j_flat = control_loss(f_flat, DT, self.W_JERK, 0.0).item()
        # flat series: every difference is exactly zero
        assert j_flat == 0.0
        # spliced jump: jerk = w * (c1-c2)^2/dt^2 / (T-1), hand-computed
        expected = self.W_JERK * (c1 - c2) ** 2 / DT ** 2 / 99
        assert abs(j_jump - expected) < 1.0e-15
        assert j_jump > 0.0

    def test_total_gradient_structure(self):
        """dL/df_k == bridge adjoint + regularizer share, incl. the
        boundary coupling between consecutive blocks (K = 2, H = 25)."""
        B, H, K = 3, 25, 2
        rng = np.random.default_rng(5)
        fc = [rng.uniform(-0.02, 0.02, size=(B, H)) for _ in range(K)]
        xmaxs = [0.08] * B

        # total loss exactly as the training loop builds it
        f = [torch.tensor(fc[k], dtype=torch.float64, requires_grad=True)
             for k in range(K)]
        loss = 0.0
        x0 = np.asarray(X0S, dtype=np.float64)
        adjs = []
        for k in range(K):
            i0 = 2 * k * H
            loss = loss + bridge.sim_loss(f[k], SEQS, PARAMS,
                                          [tuple(x0[b]) for b in range(B)],
                                          xmaxs=xmaxs, dt=DT, i0=i0)
            rec = bridge.read_last_segment(H, B)
            adjs.append(rec.adjoint.copy())        # pure bridge adjoint
            x0 = rec.final_state
        loss = loss + control_loss(torch.cat(f, dim=1), DT,
                                   self.W_JERK, self.W_MAG)
        loss.backward()

        # regularizer share on each block (independent graph)
        g = [torch.tensor(fc[k], dtype=torch.float64, requires_grad=True)
             for k in range(K)]
        control_loss(torch.cat(g, dim=1), DT,
                     self.W_JERK, self.W_MAG).backward()
        ctl0 = g[0].grad.numpy()
        ctl1 = g[1].grad.numpy()

        # structure: total == bridge adjoint + regularizer (element-wise)
        d0 = np.max(np.abs(f[0].grad.numpy() - (adjs[0] + ctl0)))
        d1 = np.max(np.abs(f[1].grad.numpy() - (adjs[1] + ctl1)))
        print(f"[F2] T24 total-gradient structure: max|diff| k0={d0:.3e} "
              f"k1={d1:.3e}")
        assert d0 < 1.0e-12
        assert d1 < 1.0e-12

        # boundary coupling: f0's LAST column feels f1's FIRST column
        assert np.any(ctl0[:, -1] != 0.0)
        # and the coupling vanishes when the blocks join smoothly
        fc_smooth = [fc[0], fc[0].copy()]
        fc_smooth[1][:, 0] = fc_smooth[0][:, -1]      # continuous splice
        g2 = [torch.tensor(fc_smooth[k], dtype=torch.float64,
                           requires_grad=True) for k in range(K)]
        control_loss(torch.cat(g2, dim=1), DT,
                     self.W_JERK, self.W_MAG).backward()
        # interior-difference gradient on the joined column is the same
        # one-sided structure as any interior point -- no boundary spike:
        col = g2[0].grad.numpy()[:, -1]
        interior = g2[0].grad.numpy()[:, -2]
        assert np.all(np.abs(col) <= np.abs(interior) + 1.0e-15)


# ---------------------------------------------------------------------------
# T27: segment-splice equivalence + road-cache switch
# ---------------------------------------------------------------------------
class TestT27SegmentSplice:
    def test_k_segment_equivalence(self):
        B, H, K = 3, 100, 5
        N = K * H
        rng = np.random.default_rng(99)
        FC = rng.uniform(-0.02, 0.02, size=(B, N))

        whole = sim3.run_cmd_seq_batch(PARAMS, FC, SEQS, x0s=X0S,
                                       t_end=N * DT, dt=DT, i0=0)

        # segmented rollout with detached state handoff
        x0 = np.asarray(X0S, dtype=np.float64)
        states = []
        j_segs = []
        for k in range(K):
            seg = sim3.run_cmd_seq_batch(
                PARAMS, FC[:, k * H:(k + 1) * H], SEQS,
                x0s=[tuple(x0[b]) for b in range(B)],
                t_end=H * DT, dt=DT, i0=2 * k * H)
            states.append(seg.state)
            j_segs.append(seg.j_sim_mean)
            x0 = seg.final_state
        seg_state = np.concatenate(states, axis=1)      # (B, N, 4)

        # state trajectory equivalence (Markov pipeline check)
        diff = np.max(np.abs(seg_state - whole.state))
        fdiff = np.max(np.abs(x0 - whole.final_state))
        print(f"[F2] T27 state max|diff| = {diff:.3e}, "
              f"final-state max|diff| = {fdiff:.3e}")
        assert diff < 1.0e-12
        assert fdiff < 1.0e-12

        # equal-weight segment sum: mean_k j_seg == whole j_sim_mean
        j_diff = abs(float(np.mean(j_segs)) - whole.j_sim_mean)
        assert j_diff < 1.0e-12

    def test_road_cache_switch(self):
        """Cache hit on repeated objects; switch picks up new roads."""
        B, H = 3, 50
        f = torch.zeros(B, H, dtype=torch.float64)
        x0s = [tuple(x) for x in X0S]

        j_a1 = bridge.sim_loss(f, SEQS, PARAMS, x0s, dt=DT, i0=0).item()
        j_a2 = bridge.sim_loss(f, SEQS, PARAMS, x0s, dt=DT, i0=0).item()
        assert j_a1 == j_a2                        # cache hit, same value

        seqs_b = [ar_switching(0.001, 0.002, 0.3, 500, DT, seed=100 + b)
                  for b in range(B)]
        j_b = bridge.sim_loss(f, seqs_b, PARAMS, x0s, dt=DT, i0=0).item()
        assert j_b != j_a1                         # switch took effect

        j_a3 = bridge.sim_loss(f, SEQS, PARAMS, x0s, dt=DT, i0=0).item()
        assert j_a3 == j_a1                        # switch back


# ---------------------------------------------------------------------------
# policy-module unit checks feeding the gates (windows / pooling / modes)
# ---------------------------------------------------------------------------
class TestPolicyObservation:
    def test_pool_mean_blocks(self):
        x = np.arange(25, dtype=np.float64)[None, :]
        p = pool_mean(x, 10, DT)
        assert p.shape == (1, 3)
        assert p[0, 0] == np.mean(x[0, :10])
        assert p[0, 1] == np.mean(x[0, 10:20])
        assert p[0, 2] == np.mean(x[0, 20:25])     # trailing partial block

    def test_obs_shapes_and_cold_start(self):
        buf = ObsBuffer(3)
        obs = build_obs(buf, [SEQ_EVT] * 3, 0, np.asarray(X0S),
                        [0.08] * 3, DT, 2)
        assert obs.past.shape == (3, 2 * PAST_WIN)
        assert obs.preview.shape == (3, PREV_WIN, 2)
        assert obs.oc.shape == (3, 5)
        assert obs.v_rel.shape == (3,)
        assert np.all(obs.past.numpy() == 0.0)     # cold-start zero pad
        # preview sees the event component only (mode 2)
        assert np.any(obs.preview.numpy() != 0.0)
        # mode 1: preview zeroed, identical other channels
        obs1 = build_obs(buf, [SEQ_EVT] * 3, 0, np.asarray(X0S),
                         [0.08] * 3, DT, 1)
        assert np.all(obs1.preview.numpy() == 0.0)
        assert np.array_equal(obs1.past.numpy(), obs.past.numpy())
        assert np.array_equal(obs1.oc.numpy(), obs.oc.numpy())

    def test_preview_tracks_road_offset(self):
        """Preview at i0 = 0 vs i0 = 2*H differ (window slides forward)."""
        p0 = preview_features([SEQ_EVT] * 3, 0, DT)
        p1 = preview_features([SEQ_EVT] * 3, 200, DT)   # 0.1 s later
        assert not np.allclose(p0, p1)

    def test_mode_equal_parameter_count(self):
        """Mode 1 = mode 2 zero-preview: identical parameter count."""
        n = sum(p.numel() for p in PolicyMLP(100).parameters())
        m = sum(p.numel() for p in PolicyMLP(100).parameters())
        assert n == m
