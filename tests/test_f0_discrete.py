"""F0 gate tests: discrete road channel + composite MS reduction (plan C, T19).

tests/test_f0_discrete.py covers docs/plan_c_policy_training_dev_test_plan.md
section 4, T19:

  T19 discrete-channel cross-checks
    - half-step sampled sine vs the analytic road_eval: grid-point lookups
      are BIT-EXACT (integer grid coordinates, interpolation frac = 0);
      off-grid lookups stay within the linear-interpolation error bound;
    - trajectory level: model3 rollout on the sampled sequence vs the
      model2b analytic rollout (sine AND bump), element-wise on all
      channels, rel < 1e-9;
    - NEGATIVE-amplitude half-sine (pothole): channel values vs the
      closed-form formula (road.py's analytic BumpRoad keeps amp >= 0;
      plan_c F0: the discrete family supports signed amps directly) plus
      a rollout cross-check through a test-local unvalidated subclass.
  T19 roughness-switch AR family statistics
    - variance ratio across the sigma step matches the level ratio band;
      P2 alignment (z(0) = 0, zdot(0) = 0 exact); seed determinism.
  T19 MS reduction vs numpy (diff < 1e-12), including the event family
    and the two-component decomposition (z = z_ar + z_evt exactly).
  T19 hinge three-state + three x_max values in ONE batch
    - inside the envelope: accumulator EXACTLY zero;
    - slightly outside: matches the numpy sum of squared overshoots and
      grows monotonically as the envelope tightens (quadratic growth);
    - FD gradient agreement is covered by the three-way test below.
  T19 three-way gradient (Tape / JVP / central FD) on the (B, n_steps)
    adjoint dJ_sim_mean/dfc3 with hinge-active elements in the batch,
    rel < 1e-6; fully saturated command slots carry exact zeros.
"""

import numpy as np
import pytest
import quadrants as qd

from d4_fixture import _Seed2D  # noqa: E402  (FwdMode seed adapter, E4 pattern)
from skyhook import model3, sim3
from skyhook.params import SemiActiveParams
from skyhook.road import BumpRoad, SineRoad
from skyhook.road_seq import (
    Event,
    ar_switching,
    events_ar,
    sample_analytic,
    sample_events,
)
from skyhook.sim2b import run_semi_cmd_batch

DT = 1.0e-3
N = 1000                       # 1 s episode (plan C T_w)
B = 3
T_END = N * DT
WEIGHTS = sim3.DEFAULT_WEIGHTS

_P0 = SemiActiveParams()
PARAMS = [SemiActiveParams(gamma_sky=_P0.gamma_sky * (0.8 + 0.2 * i),
                           c_min=_P0.c_min, c_max=_P0.c_max)
          for i in range(B)]
# nonzero ICs (plan C discipline: sqrt(0) avoidance kept even though the
# MS reduction has no sqrt)
X0S = [(0.0, 0.01, 0.0, -0.02),
       (0.005, -0.01, 0.02, 0.03),
       (-0.005, 0.02, -0.01, 0.0)]

_FC_RNG = np.random.default_rng(123)
FC = _FC_RNG.uniform(-1.0, 1.0, size=(B, N))
FC[2, ::10] *= 8.0             # adversarial spikes -> saturated command slots

# mixed road families: analytic sine / switching AR / AR + signed events
EVTS = [Event(amp=-0.02, t0=0.30, width=0.08, kind="pothole"),
        Event(amp=0.015, t0=0.60, width=0.10, kind="bump")]
SEQS = [sample_analytic(SineRoad(amp=0.02, freq_hz=2.0, ramp_t=0.5), N, DT),
        ar_switching(0.003, 0.012, 0.5, N, DT, seed=7),
        events_ar(EVTS, 0.003, 0.006, 0.4, N, DT, seed=11)]


class _SignedBump(BumpRoad):
    """BumpRoad without the amp >= 0 guard (test-only analytic reference)."""

    def __init__(self, amp, t0, width):
        self.amp = float(amp)
        self.t0 = float(t0)
        self.width = float(width)


def _perturbed_fc(fc, b, n, delta):
    out = np.array(fc, dtype=np.float64, copy=True)
    out[b, n] += delta
    return out


# ---------------------------------------------------------------------------
# T19: discrete channel cross-checks
# ---------------------------------------------------------------------------
class TestT19DiscreteChannel:
    def test_sine_grid_lookup_bit_exact(self):
        """Grid-point lookups return the stored samples bit-exactly."""
        seq = SEQS[0]
        road = SineRoad(amp=0.02, freq_hz=2.0, ramp_t=0.5)
        sim3._setup_road([seq], 0, N, 1)

        grid = np.arange(0, seq.m, 2, dtype=np.float64)      # every whole step
        k = grid.size
        for i in range(k):
            model3.pr_s[i] = grid[i]
        model3.probe_road_seq(0, k)
        z_probe = model3.pr_z.to_numpy()[:k]
        zd_probe = model3.pr_zd.to_numpy()[:k]
        z_ref, zd_ref = road.eval(grid * (DT / 2.0))
        assert np.array_equal(z_probe, z_ref)
        assert np.array_equal(zd_probe, zd_ref)

    def test_sine_off_grid_interp_bound(self):
        """Off-grid linear interpolation stays within the error bound.

        |interp - true| <= (1/8) h^2 * max|z''| for position and
        (1/8) h^2 * max|z'''| for velocity (h = dt/2).
        """
        seq = SEQS[0]
        road = SineRoad(amp=0.02, freq_hz=2.0, ramp_t=0.5)
        sim3._setup_road([seq], 0, N, 1)

        # error bound (1/8) h^2 max|z''| / (1/8) h^2 max|z'''| with the
        # maxima estimated from the ANALYTIC road on a fine grid (the
        # smoothstep ramp envelope contributes to z''/z''' as well)
        h = DT / 2.0
        tf = np.linspace(0.0, T_END, 20001)
        hf = tf[1] - tf[0]
        zf, _ = road.eval(tf)
        zdd = (zf[2:] - 2.0 * zf[1:-1] + zf[:-2]) / (hf * hf)
        zddd = (zf[4:] - 2.0 * zf[3:-1] + 2.0 * zf[1:-3] - zf[:-4]) / (hf ** 3)
        z_bound = 0.125 * h * h * np.max(np.abs(zdd))
        zd_bound = 0.125 * h * h * np.max(np.abs(zddd))

        offsets = [0.25, 0.5, 0.75]
        s = np.concatenate([np.arange(0, seq.m - 1, 7) + off for off in offsets])
        k = s.size
        for i in range(k):
            model3.pr_s[i] = s[i]
        model3.probe_road_seq(0, k)
        z_ref, zd_ref = road.eval(s * (DT / 2.0))
        ez = np.abs(model3.pr_z.to_numpy()[:k] - z_ref)
        ezd = np.abs(model3.pr_zd.to_numpy()[:k] - zd_ref)
        print(f"[F0] T19 off-grid interp: max|dz|={ez.max():.3e} "
              f"(bound {z_bound:.3e}), max|dzd|={ezd.max():.3e} "
              f"(bound {zd_bound:.3e})")
        assert ez.max() < z_bound
        assert ezd.max() < zd_bound

    def test_rollout_matches_analytic(self):
        """model3 sampled-sequence rollout vs model2b analytic rollout."""
        roads = [SineRoad(amp=0.02, freq_hz=2.0, ramp_t=0.5),
                 _SignedBump(amp=0.05, t0=0.10, width=0.20)]
        seqs = [sample_analytic(r, N, DT) for r in roads]
        x0s = X0S[:2]
        fc = FC[:2]
        ref = run_semi_cmd_batch(PARAMS[:2], fc, roads, x0s=x0s,
                                 t_end=T_END, dt=DT)
        out = sim3.run_cmd_seq_batch(PARAMS[:2], fc, seqs, x0s=x0s,
                                     t_end=T_END, dt=DT)
        diffs = {
            "state": np.max(np.abs(out.state - ref.state)),
            "a2": np.max(np.abs(out.a2 - ref.a2)),
            "fd": np.max(np.abs(out.fd - ref.fd)),
            "mask": np.max(np.abs(out.mask - ref.mask)),
            "ft": np.max(np.abs(out.ft - ref.ft)),
            "final": np.max(np.abs(out.final_state - ref.final_state)),
        }
        travel_ref = ref.state[:, :, 1] - ref.state[:, :, 0]
        diffs["travel"] = np.max(np.abs(out.travel - travel_ref))
        print("[F0] T19 rollout vs analytic: "
              + ", ".join(f"{k}={v:.3e}" for k, v in diffs.items()))
        assert all(v < 1.0e-9 for v in diffs.values())

    def test_negative_pulse_channel_and_rollout(self):
        """Signed (negative) half-sine: channel vs formula, rollout vs analytic."""
        ev = Event(amp=-0.02, t0=0.30, width=0.08, kind="pothole")
        seq = events_ar([ev], 0.0, 0.0, None, N, DT)

        # channel values vs the closed-form formula (road.py stays amp >= 0)
        tg = seq.t_grid
        inside = (tg >= ev.t0) & (tg <= ev.t0 + ev.width)
        ph = np.clip(np.pi * (tg - ev.t0) / ev.width, 0.0, np.pi)
        z_ref = np.where(inside, ev.amp * np.sin(ph), 0.0)
        zd_ref = np.where(inside, ev.amp * (np.pi / ev.width) * np.cos(ph), 0.0)
        assert np.max(np.abs(seq.z - z_ref)) < 1.0e-15
        assert np.max(np.abs(seq.zd - zd_ref)) < 1.0e-15
        # pure-event road: AR component identically zero, decomposition exact
        assert np.all(seq.z_ar == 0.0)
        assert np.array_equal(seq.z, seq.z_evt)

        # rollout cross-check through the unvalidated analytic subclass
        road = _SignedBump(amp=ev.amp, t0=ev.t0, width=ev.width)
        ref = run_semi_cmd_batch(PARAMS[:1], FC[:1], [road], x0s=X0S[:1],
                                 t_end=T_END, dt=DT)
        out = sim3.run_cmd_seq_batch(PARAMS[:1], FC[:1], [seq], x0s=X0S[:1],
                                     t_end=T_END, dt=DT)
        diff = max(np.max(np.abs(out.state - ref.state)),
                   np.max(np.abs(out.a2 - ref.a2)),
                   np.max(np.abs(out.fd - ref.fd)),
                   np.max(np.abs(out.mask - ref.mask)),
                   np.max(np.abs(out.ft - ref.ft)),
                   np.max(np.abs(out.final_state - ref.final_state)))
        print(f"[F0] T19 negative-pulse rollout vs analytic: diff={diff:.3e}")
        assert diff < 1.0e-9


# ---------------------------------------------------------------------------
# T19: roughness-switch AR family statistics
# ---------------------------------------------------------------------------
class TestT19ArSwitch:
    def test_alignment_and_determinism(self):
        seq = ar_switching(0.003, 0.012, 0.5, N, DT, seed=2026)
        assert seq.z[0] == 0.0            # P2: z(0) = 0 exactly
        assert seq.zd[0] == 0.0           # P2: zdot(0) = 0 exactly
        assert np.all(seq.z_evt == 0.0)   # pure AR family: no events
        assert np.array_equal(seq.z, seq.z_ar)

        again = ar_switching(0.003, 0.012, 0.5, N, DT, seed=2026)
        other = ar_switching(0.003, 0.012, 0.5, N, DT, seed=2027)
        assert np.array_equal(seq.z, again.z) and np.array_equal(seq.zd, again.zd)
        assert not np.array_equal(seq.z, other.z)

    def test_sigma_step_variance_ratio(self):
        """Variance jumps across the sigma step; levels match the targets."""
        n4, t_s = 4000, 2.0
        s1, s2 = 0.003, 0.012
        seq = ar_switching(s1, s2, t_s, n4, DT, seed=2026)
        tg = seq.t_grid
        w_before = (tg >= 0.2) & (tg < t_s - 0.2)
        w_after = (tg >= t_s + 0.2) & (tg <= 4.0 - 0.2)
        v1, v2 = seq.z[w_before].var(), seq.z[w_after].var()
        ratio = v2 / v1
        print(f"[F0] T19 AR switch: std_before={np.sqrt(v1):.5f} "
              f"(target {s1}), std_after={np.sqrt(v2):.5f} (target {s2}), "
              f"var ratio={ratio:.2f} (target {(s2 / s1) ** 2:.1f})")
        # effective sample size ~ window / (2*tau_int) makes the estimates
        # loose; with the fixed seed a wide band around the 16x target
        assert 0.6 * s1 < np.sqrt(v1) < 1.5 * s1
        assert 0.6 * s2 < np.sqrt(v2) < 1.5 * s2
        assert 8.0 < ratio < 26.0
        assert np.max(np.abs(seq.z)) < 6.0 * s2

    def test_event_sampler_constraints(self):
        rng = np.random.default_rng(99)
        worst = {"width": 1.0, "gap": 1.0}
        for _ in range(50):
            evs = sample_events(rng, N, DT)
            assert 1 <= len(evs) <= 3
            for i, ev in enumerate(evs):
                assert ev.width >= 0.05 - 1e-12
                assert 0.10 - 1e-9 <= ev.t0
                assert ev.t0 + ev.width <= N * DT - 0.10 + 1e-9
                if i > 0:
                    gap = ev.t0 - (evs[i - 1].t0 + evs[i - 1].width)
                    worst["gap"] = min(worst["gap"], gap)
            worst["width"] = min(worst["width"], min(e.width for e in evs))
        print(f"[F0] T19 event sampler: min width={worst['width']:.3f}, "
              f"min gap={worst['gap']:.3f}")
        assert worst["gap"] >= 0.15 - 1e-9

    def test_events_ar_decomposition(self):
        """z = z_ar + z_evt exactly; the event component is the pulse sum."""
        seq = SEQS[2]
        assert np.array_equal(seq.z, seq.z_ar + seq.z_evt)
        z_evt_ref = np.zeros(seq.m)
        zd_evt_ref = np.zeros(seq.m)
        for ev in EVTS:
            ze, zde = ev.eval(seq.t_grid)
            z_evt_ref += ze
            zd_evt_ref += zde
        assert np.allclose(seq.z_evt, z_evt_ref, rtol=0, atol=1e-15)
        assert np.all(np.isfinite(seq.zd))
        # P2 alignment holds for the mixed family too (AR offset + events
        # placed away from t = 0)
        assert seq.z[0] == 0.0 and seq.zd[0] == 0.0


# ---------------------------------------------------------------------------
# T19: composite MS reduction vs numpy
# ---------------------------------------------------------------------------
class TestT19MsReduction:
    def test_ms_vs_numpy(self):
        out = sim3.run_cmd_seq_batch(PARAMS, FC, SEQS, x0s=X0S,
                                     t_end=T_END, dt=DT)
        ov = np.maximum(np.abs(out.travel) - sim3.X_MAX_DEFAULT, 0.0)
        terms_ref = np.stack([
            np.mean(out.a2 ** 2, axis=1),
            np.mean(out.travel ** 2, axis=1),
            np.mean(out.ft ** 2, axis=1),
            np.mean(ov ** 2, axis=1),
        ], axis=1)
        scale = np.maximum(np.abs(terms_ref).max(), 1e-30)
        diff_terms = np.max(np.abs(out.j_terms - terms_ref))
        j_ref = terms_ref @ np.array(WEIGHTS)
        diff_j = np.max(np.abs(out.j_sim - j_ref))
        diff_mean = abs(out.j_sim_mean - float(np.mean(j_ref)))
        print(f"[F0] T19 MS reduction: diff_terms={diff_terms:.3e} "
              f"diff_j={diff_j:.3e} diff_mean={diff_mean:.3e} "
              f"(scale {scale:.3e})")
        assert diff_terms / scale < 1.0e-12
        assert diff_j / max(abs(j_ref).max(), 1e-30) < 1.0e-12
        assert diff_mean / max(abs(np.mean(j_ref)), 1e-30) < 1.0e-12
        # default envelope 0.08 m: travel stays inside -> hinge exactly 0
        assert np.all(out.j_terms[:, 3] == 0.0)


# ---------------------------------------------------------------------------
# T19: hinge three-state, three x_max values in one batch
# ---------------------------------------------------------------------------
class TestT19Hinge:
    def test_three_states_one_batch(self):
        # measure the stroke envelope with a huge x_max first
        big = sim3.run_cmd_seq_batch(PARAMS, FC, SEQS, x0s=X0S,
                                     xmaxs=[10.0] * B, t_end=T_END, dt=DT)
        dmax = np.max(np.abs(big.travel), axis=1)
        assert np.all(dmax > 1e-3)  # sanity: the excitation moves the stroke

        xmaxs = [10.0, 0.9 * dmax[1], 0.3 * dmax[2]]
        out = sim3.run_cmd_seq_batch(PARAMS, FC, SEQS, x0s=X0S, xmaxs=xmaxs,
                                     t_end=T_END, dt=DT)

        # state 1: inside the envelope -> accumulator EXACTLY zero
        assert out.j_terms[0, 3] == 0.0

        # states 2/3: slightly / deeply outside -> numpy formula match
        for b in (1, 2):
            ov = np.maximum(np.abs(out.travel[b]) - xmaxs[b], 0.0)
            ref = float(np.mean(ov ** 2))
            assert ref > 0.0
            assert abs(out.j_terms[b, 3] - ref) / ref < 1.0e-12
        assert out.j_terms[2, 3] > out.j_terms[1, 3] > 0.0

        # quadratic growth: tightening the envelope grows the term monotonically
        # and the numpy match keeps holding (sum of squared overshoots)
        xmaxs_tight = [10.0, 0.8 * dmax[1], 0.2 * dmax[2]]
        tight = sim3.run_cmd_seq_batch(PARAMS, FC, SEQS, x0s=X0S,
                                       xmaxs=xmaxs_tight, t_end=T_END, dt=DT)
        for b in (1, 2):
            ov = np.maximum(np.abs(tight.travel[b]) - xmaxs_tight[b], 0.0)
            ref = float(np.mean(ov ** 2))
            assert abs(tight.j_terms[b, 3] - ref) / ref < 1.0e-12
        assert tight.j_terms[1, 3] > out.j_terms[1, 3]
        assert tight.j_terms[2, 3] > out.j_terms[2, 3]
        print(f"[F0] T19 hinge: dmax={dmax}, "
              f"j_lim(0.9x)={out.j_terms[1, 3]:.3e}, "
              f"j_lim(0.8x)={tight.j_terms[1, 3]:.3e}, "
              f"j_lim(0.3x)={out.j_terms[2, 3]:.3e}, "
              f"j_lim(0.2x)={tight.j_terms[2, 3]:.3e}")


# ---------------------------------------------------------------------------
# T19: three-way gradient (Tape / JVP / FD) with hinge-active elements
# ---------------------------------------------------------------------------
class TestT19ThreeWayGradient:
    def _batch_with_hinge(self):
        """Fixture: xmaxs chosen so element 0 stays inside, 1 slightly and
        2 deeply outside the stroke envelope (hinge-active J_sim)."""
        big = sim3.run_cmd_seq_batch(PARAMS, FC, SEQS, x0s=X0S,
                                     xmaxs=[10.0] * B, t_end=T_END, dt=DT)
        dmax = np.max(np.abs(big.travel), axis=1)
        return [10.0, 0.9 * dmax[1], 0.3 * dmax[2]]

    def test_three_way(self):
        xmaxs = self._batch_with_hinge()
        out = sim3.run_cmd_seq_batch(PARAMS, FC, SEQS, x0s=X0S, xmaxs=xmaxs,
                                     t_end=T_END, dt=DT)
        assert out.j_terms[1, 3] > 0.0 and out.j_terms[2, 3] > 0.0

        # reverse Tape: the whole (B_MAX, T_B_MAX) adjoint dJ_sim_mean/dfc3
        sim3._setup(PARAMS, X0S, xmaxs)
        sim3._setup_road(SEQS, 0, N, B)
        sim3._setup_fcmd(FC, N, B)
        with qd.ad.Tape(loss=model3.j_mean3):
            sim3.run_jsim_chain(N, DT, B, 0, WEIGHTS)
        grad = model3.fc3.grad.to_numpy()
        g = grad[:B, :N]

        # untouched slots carry exactly zero
        assert np.all(grad[:, N:] == 0.0)
        assert np.all(grad[B:, :] == 0.0)

        def pick(pred):
            for b in range(B):
                for n in range(N):
                    if pred(b, n):
                        return b, n
            return None

        slots = {
            "inband": pick(lambda b, n: out.mask[b, n] == 0.0 and g[b, n] != 0.0),
            "hinge_active": pick(lambda b, n: b == 1 and out.mask[b, n] == 0.0
                                 and g[b, n] != 0.0),
            "saturated": pick(lambda b, n: out.mask[b, n] == 1.0
                              and g[b, n] == 0.0),
        }
        assert slots["inband"] is not None
        assert slots["hinge_active"] is not None
        assert slots["saturated"] is not None

        rel = lambda x, y: abs(x - y) / max(abs(y), 1e-300)  # noqa: E731
        for name, (b, n) in slots.items():
            seed = np.zeros((model3.B_MAX, model3.T_B_MAX), dtype=np.float64)
            seed[b, n] = 1.0
            sim3._setup(PARAMS, X0S, xmaxs)
            sim3._setup_fcmd(FC, N, B)
            with qd.ad.FwdMode(loss=model3.j_mean3, param=model3.fc3,
                               seed=seed.view(_Seed2D)):
                sim3.run_jsim_chain(N, DT, B, 0, WEIGHTS)
            jvp = model3.j_mean3.dual[None]

            # central FD (roads already on the device; only fc changes)
            def jmean_of(fc):
                sim3._setup(PARAMS, X0S, xmaxs)
                sim3._setup_fcmd(fc, N, B)
                sim3.run_jsim_chain(N, DT, B, 0, WEIGHTS)
                return model3.j_mean3[None]

            h = 1.0e-4
            fd = (jmean_of(_perturbed_fc(FC, b, n, +h))
                  - jmean_of(_perturbed_fc(FC, b, n, -h))) / (2.0 * h)

            print(f"[F0] T19 slot {name} (b={b}, n={n}): AD={g[b, n]:.10e} "
                  f"JVP={jvp:.10e} FD={fd:.10e} rel_AD_FD={rel(g[b, n], fd):.3e} "
                  f"rel_AD_JVP={rel(g[b, n], jvp):.3e}")
            if name == "saturated":
                assert g[b, n] == 0.0 and jvp == 0.0 and fd == 0.0
            else:
                assert rel(g[b, n], fd) < 1.0e-6
                assert rel(g[b, n], jvp) < 1.0e-6
