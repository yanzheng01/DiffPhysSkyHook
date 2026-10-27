"""D1 passive-baseline tests (Gate: T1 frequency response, T2 modal frequencies,
T3 energy audit), plus road kernel/python cross-check and record-channel checks.

Stage D1 of docs/plan_a_dev_test_plan.md: passive quarter-car
(gamma_sky = 0) with Quadrants RK4 backend, dt = 1 ms, float64.

Anchors:
  T1  steady-state amplitudes vs analytic transfer functions |X1/Z|, |X2/Z|
      (initial state set to the analytic steady state, so the response is
      transient-free; amplitudes fitted by single-frequency least squares);
      tolerance: relative amplitude error < 1e-3.
  T2  free-decay FFT peaks vs analytic coupled undamped eigenfrequencies
      (the coupled eigenproblem, not the decoupled omega_s / omega_t, is the
      honest anchor at mu = 0.1); tolerance: deviation < 2%.
  T3  mechanical energy of the passive system decays monotonically, and the
      decay matches the analytic dissipation integral; dE <= 0 within tol.
"""

import math

import numpy as np
import pytest

from analytic import (coupled_eigen, fft_peak_freq, fit_amplitude,
                      mechanical_energy, transfer_functions)
from skyhook.params import PassiveParams
from skyhook.road import BumpRoad, SineRoad, ZeroRoad
from skyhook.sim import run_passive

P = PassiveParams()
DT = 1.0e-3


# ---------------------------------------------------------------------------
# helpers (shared analytic anchors, tests/analytic.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# road cross-check: kernel road_eval vs python Road.eval
# ---------------------------------------------------------------------------
class TestRoadKernelMatch:
    @pytest.mark.parametrize("road", [
        ZeroRoad(),
        SineRoad(0.05, 3.0, ramp_t=0.5),
        SineRoad(0.05, 3.0, ramp_t=0.0),
        BumpRoad(0.05, t0=1.0, width=0.2),
    ], ids=["zero", "sine-ramp", "sine-noramp", "bump"])
    def test_kernel_matches_python(self, road):
        from skyhook import model

        rng = np.random.default_rng(0)
        ts = np.concatenate([
            np.linspace(0.0, 2.0, 100),
            rng.uniform(0.0, 2.0, 100),
            np.array([0.0, 0.25, 0.5, 1.0, 1.2, 1.0 + 1.0e-9]),
        ])
        n = len(ts)
        assert n <= model.t_probe.shape[0]
        for i, tv in enumerate(ts):
            model.t_probe[i] = tv
        model.probe_road(n, *road.kernel_args())
        z_k = model.z_probe.to_numpy()[:n]
        zd_k = model.zd_probe.to_numpy()[:n]
        z_py, zd_py = road.eval(ts)
        err_z = np.max(np.abs(z_k - z_py))
        err_zd = np.max(np.abs(zd_k - zd_py))
        print(f"[D1] road {type(road).__name__}: max |dz|={err_z:.3e}, "
              f"max |dzdot|={err_zd:.3e}")
        assert np.allclose(z_k, z_py, rtol=1e-12, atol=1e-14)
        assert np.allclose(zd_k, zd_py, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# record-channel consistency (whole-step v2dot and F_d vs numpy formula)
# ---------------------------------------------------------------------------
class TestRecordChannels:
    def test_a2_and_fd_channels(self):
        road = SineRoad(amp=0.02, freq_hz=2.0, ramp_t=0.5)
        res = run_passive(P, road, x0=(0.0, 0.0, 0.0, 0.0), t_end=3.0, dt=DT)
        x1, x2, v1, v2 = (res.state[:, i] for i in range(4))
        a2_ref = -(P.omega_s_sq * (x2 - x1) + P.gamma_s * (v2 - v1))
        fd_ref = P.gamma_s * (v2 - v1)
        err_a2 = np.max(np.abs(res.a2 - a2_ref))
        err_fd = np.max(np.abs(res.fd - fd_ref))
        print(f"[D1] record channels: max |a2 err|={err_a2:.3e}, "
              f"max |fd err|={err_fd:.3e}")
        assert err_a2 < 1e-15
        assert err_fd < 1e-15

    def test_bump_forward_smoke(self):
        road = BumpRoad(amp=0.05, t0=1.0, width=0.2)
        res = run_passive(P, road, t_end=3.0, dt=DT)
        assert np.all(np.isfinite(res.state))
        travel = res.state[:, 1] - res.state[:, 0]
        peak = np.max(np.abs(travel))
        print(f"[D1] bump smoke: peak |x2-x1| = {peak:.4e}")
        assert peak > 0.0


# ---------------------------------------------------------------------------
# T1: analytic frequency response vs simulated steady-state amplitude
# ---------------------------------------------------------------------------
class TestT1FrequencyResponse:
    def test_steady_state_amplitudes(self):
        w_lo, w_hi, _, _ = coupled_eigen(P)
        f_lo, f_hi = w_lo / (2 * math.pi), w_hi / (2 * math.pi)
        freqs = np.unique(np.clip(np.concatenate([
            np.geomspace(0.5, 20.0, 17),
            f_lo * np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25]),
            f_hi * np.array([0.85, 0.95, 1.0, 1.05, 1.15]),
        ]), 0.3, 30.0))

        A = 1.0
        worst = {"x1": 0.0, "x2": 0.0}
        for f in freqs:
            omega = 2.0 * math.pi * f
            h1, h2 = transfer_functions(omega, P)
            X1, X2 = h1 * (-1j * A), h2 * (-1j * A)  # z = A sin(omega t)
            x0 = (X1.real, X2.real, -omega * X1.imag, -omega * X2.imag)
            road = SineRoad(amp=A, freq_hz=f, ramp_t=0.0)
            t_end = max(4.0 / f, 6.0)
            res = run_passive(P, road, x0=x0, t_end=t_end, dt=DT)
            m = len(res.t) // 2  # fit the late half: transient-free window
            amp1 = fit_amplitude(res.t[m:], res.state[m:, 0], omega)
            amp2 = fit_amplitude(res.t[m:], res.state[m:, 1], omega)
            for key, amp, x_ref in (("x1", amp1, abs(X1)), ("x2", amp2, abs(X2))):
                denom = max(abs(x_ref), 1e-6 * A)
                worst[key] = max(worst[key], abs(amp - abs(x_ref)) / denom)
        print(f"[D1] T1 worst relative amplitude error: "
              f"|X1/Z| {worst['x1']:.3e}, |X2/Z| {worst['x2']:.3e} "
              f"over {len(freqs)} frequencies")
        assert worst["x1"] < 1e-3
        assert worst["x2"] < 1e-3


# ---------------------------------------------------------------------------
# T2: modal frequencies from free-decay FFT vs analytic coupled eigenvalues
# ---------------------------------------------------------------------------
@pytest.fixture(scope="class")
def body_mode_decay():
    """Free decay of the body mode (shared by T2 and T3)."""
    w_lo, _, r_lo, _ = coupled_eigen(P)
    x0 = (1.0e-3, r_lo * 1.0e-3, 0.0, 0.0)
    return run_passive(P, ZeroRoad(), x0=x0, t_end=16.0, dt=DT)


class TestT2ModalFrequencies:
    def test_body_mode(self, body_mode_decay):
        w_lo, _, _, _ = coupled_eigen(P)
        f_ref = w_lo / (2.0 * math.pi)
        f_meas = fft_peak_freq(body_mode_decay.t, body_mode_decay.state[:, 1], f_ref)
        dev = abs(f_meas / f_ref - 1.0)
        print(f"[D1] T2 body mode: FFT peak {f_meas:.4f} Hz vs analytic "
              f"{f_ref:.4f} Hz, deviation {dev * 100:.2f}%")
        assert dev < 0.02

    def test_wheel_mode(self):
        _, w_hi, _, r_hi = coupled_eigen(P)
        f_ref = w_hi / (2.0 * math.pi)
        x0 = (1.0e-3, r_hi * 1.0e-3, 0.0, 0.0)
        res = run_passive(P, ZeroRoad(), x0=x0, t_end=16.0, dt=DT)
        f_meas = fft_peak_freq(res.t, res.state[:, 0], f_ref)
        dev = abs(f_meas / f_ref - 1.0)
        print(f"[D1] T2 wheel mode: FFT peak {f_meas:.4f} Hz vs analytic "
              f"{f_ref:.4f} Hz, deviation {dev * 100:.2f}%")
        assert dev < 0.02


# ---------------------------------------------------------------------------
# T3: energy audit on the passive system (zero road)
# ---------------------------------------------------------------------------
class TestT3EnergyAudit:
    def test_energy_monotone_decay(self, body_mode_decay):
        E = mechanical_energy(body_mode_decay.state, P)
        de = np.diff(E)
        E0 = E[0]
        print(f"[D1] T3 energy: E0={E0:.6e}, E_end={E[-1]:.6e}, "
              f"max step dE={de.max():.3e} (tol {1e-10 * E0:.1e})")
        assert de.max() <= 1e-10 * E0  # monotone within tolerance
        assert E[-1] < 0.5 * E0        # substantial decay actually happened

    def test_dissipation_balance(self, body_mode_decay):
        res = body_mode_decay
        v1, v2 = res.state[:, 2], res.state[:, 3]
        diss = P.gamma_s * (v2 - v1) ** 2 + P.mu * P.gamma_t * v1 ** 2
        W = np.trapezoid(diss, res.t)
        E = mechanical_energy(res.state, P)
        balance_err = abs((E[0] - E[-1]) - W) / E[0]
        print(f"[D1] T3 dissipation balance: E0-E_end={E[0] - E[-1]:.6e}, "
              f"integral={W:.6e}, rel err={balance_err:.3e}")
        assert balance_err < 1e-4
