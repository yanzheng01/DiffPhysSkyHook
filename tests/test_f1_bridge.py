"""F1 gate tests (plan C section 4): torch <-> Quadrants bridge.

  T20 bridge adjoint consistency: the (B, H) gradient torch receives from
     bridge.backward == the adjoint read directly from fc3.grad after the
     same Tape run (same decay factor), element-wise < 1e-12; the forward
     J also matches the Tape loss bit-close.
  T21 torch end-to-end linear probe: f_cmd = W @ h (fixed h), the bridge
     gradient dJ/dW vs the hand-computed outer(dJ/df, h) and vs central
     finite differences on sampled in-band slots, rel < 1e-5.
  T22 write-performance spike (diagnostic soft gate): steady-state
     per-segment bridge call cost, extrapolated to K segments x 50
     iterations; recorded in the log, with a loose regression bound.
"""

import time

import numpy as np
import pytest
import torch
import quadrants as qd

from skyhook import bridge, model3, sim3
from skyhook.params import SemiActiveParams
from skyhook.road import SineRoad
from skyhook.road_seq import (
    Event,
    ar_switching,
    events_ar,
    sample_analytic,
)

DT = 1.0e-3
H = 100                          # segment steps (plan C decision 1: H = 0.1 s)
N = 1000                         # whole-episode steps (road grid coverage)
B = 3
I0 = 2 * 5 * H                   # segment 6 of the episode (nonzero offset)
WEIGHTS = sim3.DEFAULT_WEIGHTS

_P0 = SemiActiveParams()
PARAMS = [SemiActiveParams(gamma_sky=_P0.gamma_sky * (0.8 + 0.2 * i),
                           c_min=_P0.c_min, c_max=_P0.c_max)
          for i in range(B)]
# nonzero ICs (plan C discipline)
X0S = [(0.0, 0.01, 0.0, -0.02),
       (0.005, -0.01, 0.02, 0.03),
       (-0.005, 0.02, -0.01, 0.0)]

_FC_RNG = np.random.default_rng(321)
FC_SEG = _FC_RNG.uniform(-1.0, 1.0, size=(B, H))

# mixed road families over the WHOLE episode grid (bridge reads via i0)
EVTS = [Event(amp=-0.02, t0=0.30, width=0.08, kind="pothole"),
        Event(amp=0.015, t0=0.60, width=0.10, kind="bump")]
SEQS = [sample_analytic(SineRoad(amp=0.02, freq_hz=2.0, ramp_t=0.5), N, DT),
        ar_switching(0.003, 0.012, 0.5, N, DT, seed=7),
        events_ar(EVTS, 0.003, 0.006, 0.4, N, DT, seed=11)]

# hinge-active envelope mix (T19 pattern: inside / slightly / deeply outside)
_probe = sim3.run_cmd_seq_batch(PARAMS, FC_SEG, SEQS, x0s=X0S,
                                xmaxs=[10.0] * B, t_end=H * DT, dt=DT, i0=I0)
_dmax = np.max(np.abs(_probe.travel), axis=1)
XMAXS = [10.0, 0.9 * _dmax[1], 0.3 * _dmax[2]]
assert _dmax[1] > 0.0 and _dmax[2] > 0.0        # nonzero travel in segment


def _tape_reference(fc, i0):
    """Direct Tape run on the same setup; returns (J, dJ/dfc (B, H))."""
    sim3._setup(PARAMS, X0S, XMAXS)
    sim3._setup_road(SEQS, i0, H, B)
    sim3._setup_fcmd(fc, H, B)
    with qd.ad.Tape(loss=model3.j_mean3):
        sim3.run_jsim_chain(H, DT, B, i0, WEIGHTS)
    j = float(model3.j_mean3[None])
    g = model3.fc3.grad.to_numpy()[:B, :H].copy()
    return j, g


# ---------------------------------------------------------------------------
# T20: bridge adjoint == direct Tape read (same decay)
# ---------------------------------------------------------------------------
class TestT20BridgeAdjoint:
    @pytest.mark.parametrize("decay", [1.0, 0.9])
    def test_adjoint_matches_tape(self, decay):
        j_ref, g_ref = _tape_reference(FC_SEG, I0)

        f = torch.tensor(FC_SEG, dtype=torch.float64, requires_grad=True)
        loss = bridge.sim_loss(f, SEQS, PARAMS, X0S, xmaxs=XMAXS,
                               weights=WEIGHTS, dt=DT, i0=I0, decay=decay)
        loss.backward()

        # forward value agreement
        assert abs(loss.item() - j_ref) < 1.0e-12

        expected = g_ref
        if decay != 1.0:
            expected = expected * (decay ** np.arange(H))[np.newaxis, :]
        diff = np.max(np.abs(f.grad.numpy() - expected))
        print(f"[F1] T20 decay={decay}: max|bridge - tape| = {diff:.3e}")
        assert diff < 1.0e-12
        assert np.any(f.grad.numpy() != 0.0)   # adjoint is not all-zero

    def test_repeated_forward_no_cross_talk(self):
        """A second forward overwrites (not accumulates) the stored adjoint:
        two identical calls give identical gradients."""
        grads = []
        for _ in range(2):
            f = torch.tensor(FC_SEG, dtype=torch.float64, requires_grad=True)
            bridge.sim_loss(f, SEQS, PARAMS, X0S, xmaxs=XMAXS,
                            weights=WEIGHTS, dt=DT, i0=I0).backward()
            grads.append(f.grad.numpy().copy())
        assert np.array_equal(grads[0], grads[1])


# ---------------------------------------------------------------------------
# T21: torch end-to-end linear probe f_cmd = W @ h
# ---------------------------------------------------------------------------
class TestT21LinearProbe:
    D = 8
    EPS = 1.0e-4

    def test_dw_bridge_vs_hand_vs_fd(self):
        rng = np.random.default_rng(777)
        h_np = rng.normal(0.0, 1.0, size=self.D)
        w_np = rng.normal(0.0, 1.0, size=(B * H, self.D)) / np.sqrt(self.D)

        # reference adjoint via the direct Tape (decay = 1)
        f0 = (w_np @ h_np).reshape(B, H)
        _, g_ref = _tape_reference(f0, I0)
        g_flat = g_ref.reshape(B * H)

        # bridge gradient through torch autograd
        h = torch.tensor(h_np, dtype=torch.float64)
        W = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
        f = (W @ h).view(B, H)
        loss = bridge.sim_loss(f, SEQS, PARAMS, X0S, xmaxs=XMAXS,
                               weights=WEIGHTS, dt=DT, i0=I0)
        loss.backward()

        # hand computation: dJ/dW = outer(dJ/df, h)
        hand = np.outer(g_flat, h_np)
        w_grad = W.grad.numpy()
        rel = np.max(np.abs(w_grad - hand)
                     / np.maximum(np.abs(hand), 1.0e-300))
        print(f"[F1] T21 dJ/dW bridge vs hand: max rel = {rel:.3e}")
        assert rel < 1.0e-5

        # central FD on sampled in-band slots (mask == 0, nonzero adjoint)
        probe = sim3.run_cmd_seq_batch(PARAMS, f0, SEQS, x0s=X0S,
                                       xmaxs=XMAXS, t_end=H * DT, dt=DT,
                                       i0=I0)
        inband = [(b, n) for b in range(B) for n in range(H)
                  if probe.mask[b, n] == 0.0 and g_ref[b, n] != 0.0]
        assert inband, "no in-band slot with nonzero adjoint"
        picks = [inband[i] for i in np.linspace(0, len(inband) - 1,
                                                num=5).astype(int)]

        def jmean_of_w(w):
            fc = (w @ h_np).reshape(B, H)
            out = sim3.run_cmd_seq_batch(PARAMS, fc, SEQS, x0s=X0S,
                                         xmaxs=XMAXS, t_end=H * DT, dt=DT,
                                         i0=I0)
            return out.j_sim_mean

        rel_fn = lambda x, y: abs(x - y) / max(abs(y), 1.0e-300)  # noqa: E731
        for b, n in picks:
            i = b * H + n
            for j in (0, self.D // 2):
                wp = w_np.copy(); wp[i, j] += self.EPS
                wm = w_np.copy(); wm[i, j] -= self.EPS
                fd = (jmean_of_w(wp) - jmean_of_w(wm)) / (2.0 * self.EPS)
                ad = hand[i, j]
                print(f"[F1] T21 FD slot (b={b}, n={n}, j={j}): "
                      f"bridge={ad:.10e} FD={fd:.10e} rel={rel_fn(ad, fd):.3e}")
                assert rel_fn(ad, fd) < 1.0e-5


# ---------------------------------------------------------------------------
# T22: write-performance spike (diagnostic soft gate)
# ---------------------------------------------------------------------------
class TestT22WritePerformance:
    def test_steady_state_segment_cost(self):
        """One JIT-warming call, then time K = 10 bridge segments (each a
        full bridge call: setup + road rewrite + fc write + Tape). Loose
        regression bound only; the numbers are the deliverable."""
        f = torch.tensor(FC_SEG, dtype=torch.float64)

        # warm-up (JIT compilation of the whole chain)
        t0 = time.perf_counter()
        bridge.sim_loss(f, SEQS, PARAMS, X0S, xmaxs=XMAXS, dt=DT, i0=0)
        t_warm = time.perf_counter() - t0

        K = 10
        t0 = time.perf_counter()
        for k in range(K):
            bridge.sim_loss(f, SEQS, PARAMS, X0S, xmaxs=XMAXS, dt=DT,
                            i0=2 * k * H)
        t_total = time.perf_counter() - t0
        per_seg = t_total / K
        n_iter = 50
        print(f"[F1] T22 warm-up (JIT): {t_warm:.3f} s | "
              f"steady-state bridge segment: {per_seg * 1e3:.2f} ms | "
              f"K={K} segments: {t_total * 1e3:.1f} ms | "
              f"{n_iter}-iteration budget extrapolation: "
              f"{per_seg * K * n_iter:.1f} s")

        # soft gate: a full bridge segment (incl. whole-episode road
        # rewrite + element-wise fc write + Tape) must stay well below
        # half a second at steady state
        assert per_seg < 0.5
