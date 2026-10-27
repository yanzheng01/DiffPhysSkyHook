"""Analytic anchors shared by D-stage tests (independent math path)."""

import math

import numpy as np


def coupled_eigen(p):
    """Analytic undamped coupled eigenfrequencies (rad/s) + eigvec ratios.

    det| (u - omega_t^2 - omega_s^2/mu)   omega_s^2/mu        |
        | omega_s^2                        (u - omega_s^2)    | = 0
    Eigenvalue ratios: X2/X1 = omega_s^2 / (omega_s^2 - u).
    """
    a = p.omega_t_sq + p.omega_s_sq / p.mu
    b = p.omega_s_sq
    disc = math.sqrt((a - b) ** 2 + 4.0 * p.omega_s_sq ** 2 / p.mu)
    u_lo = 0.5 * (a + b - disc)  # body mode
    u_hi = 0.5 * (a + b + disc)  # wheel-hop mode
    r_lo = p.omega_s_sq / (p.omega_s_sq - u_lo)  # X2/X1
    r_hi = p.omega_s_sq / (p.omega_s_sq - u_hi)
    return math.sqrt(u_lo), math.sqrt(u_hi), r_lo, r_hi


def transfer_functions(omega: float, p):
    """Analytic passive TFs X1/Z, X2/Z at excitation frequency omega (rad/s).

    From the phasor equations (m2 = 1 gauge, gamma_sky = 0):
      mu s^2 X1 + mu G_t (X1 - Z) + K (X1 - X2) = 0,  s^2 X2 + K (X2 - X1) = 0
      with G_t = omega_t^2 + s gamma_t,  K = omega_s^2 + s gamma_s,  s = i omega.
    Eliminating X1 (X1 - X2 = s^2 X2 / K from the second equation) gives
      H2 = mu K G_t / [mu (K + s^2)(G_t + s^2) + s^2 K]
    whose undamped zeros coincide with the coupled eigenfrequencies.
    """
    s = 1j * omega
    K = p.omega_s_sq + s * p.gamma_s
    Gt = p.omega_t_sq + s * p.gamma_t
    denom = p.mu * (K + s * s) * (Gt + s * s) + s * s * K
    h2 = p.mu * K * Gt / denom
    h1 = p.mu * (K + s * s) * Gt / denom
    return h1, h2


def qep_damped_roots(p):
    """Exact damped eigenvalues (complex) of the model-I 2-DOF system.

    det(M s^2 + C s + K) = 0 with
      M = diag(mu, 1)
      C = [[mu*gamma_t + gamma_s, -gamma_s], [-gamma_s, gamma_s + gamma_sky]]
      K = [[mu*omega_t^2 + omega_s^2, -omega_s^2], [-omega_s^2, omega_s^2]]
    """
    c1 = p.mu * p.gamma_t
    c2 = p.gamma_s
    k1 = p.mu * p.omega_t_sq
    k2 = p.omega_s_sq
    a1 = [p.mu, c1 + c2, k1 + k2]          # mu s^2 + (c1+c2) s + (k1+k2)
    a2 = [1.0, c2 + p.gamma_sky, k2]       # s^2 + (c2+gamma_sky) s + k2
    det = np.polymul(a1, a2)
    bb = np.polymul([c2, k2], [c2, k2])    # (c2 s + k2)^2
    det[-len(bb):] -= bb
    return np.roots(det)


def fit_amplitude(t: np.ndarray, x: np.ndarray, omega: float) -> float:
    """Single-frequency LSQ amplitude of x ~ a cos(wt) + b sin(wt)."""
    M = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
    coef, *_ = np.linalg.lstsq(M, x, rcond=None)
    return math.hypot(coef[0], coef[1])


def fft_peak_freq(t: np.ndarray, x: np.ndarray, f_guess: float) -> float:
    """FFT peak frequency with Hann window, 8x zero pad, log-parabolic interp."""
    n = len(x)
    dt = t[1] - t[0]
    xw = x * np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    mag = np.abs(np.fft.rfft(xw, nfft)) + 1e-300
    freqs = np.fft.rfftfreq(nfft, dt)
    band = np.where((freqs > 0.5 * f_guess) & (freqs < 1.8 * f_guess))[0]
    k = band[np.argmax(mag[band])]
    if 0 < k < len(mag) - 1:
        y0, y1, y2 = np.log(mag[k - 1]), np.log(mag[k]), np.log(mag[k + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 0.0:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) <= 1.0:
                k = k + delta
    return k * freqs[1]


def envelope_sigma(t: np.ndarray, x: np.ndarray, fit_lo: float, fit_hi: float) -> float:
    """Decay rate sigma (1/s) from the Hilbert-transform envelope log-regression."""
    from scipy.signal import hilbert

    env = np.abs(hilbert(x))
    m = (t >= fit_lo) & (t <= fit_hi)
    slope = np.polyfit(t[m], np.log(env[m]), 1)[0]
    return -slope


def fit_mode_sigma(t: np.ndarray, x: np.ndarray, sigma0: float,
                   omega0: float) -> float:
    """Decay rate of a single damped mode via direct nonlinear LSQ.

    Fits x ~ a*exp(-sigma*t)*cos(omega*t + phi). For an exact single-mode
    response this is bias-free, unlike the Hilbert envelope (whose analytic
    signal leaks across the sigma/omega-scaled spectral overlap and record
    edges). Used by T4 where the IC is an exact damped-eigenvector excitation.
    """
    from scipy.optimize import curve_fit

    def model(t_: np.ndarray, a, sigma, omega, phi):
        return a * np.exp(-sigma * t_) * np.cos(omega * t_ + phi)

    popt, _ = curve_fit(model, t, x,
                        p0=[max(abs(x[0]), 1e-30), sigma0, omega0, 0.0],
                        maxfev=20000)
    return popt[1]


def body_mode_ic(p, amp: float = 1.0e-3):
    """Exact single-mode initial state x(t) = Re(amp * phi * e^{s t}).

    s is the damped body-mode root (upper-half-plane pair member with the
    lower frequency) of the model-I QEP; the eigenvector follows the row-2
    eigenrelation
        phi1 = (s^2 + (gamma_s + gamma_sky) s + omega_s^2) / (gamma_s s + omega_s^2)
    (denominator gamma_s*s + omega_s^2, NOT omega_s^2 -- dividing by omega_s^2
    alone is only valid for gamma_s = 0 and contaminates the IC with the
    wheel-hop mode, which floors the RK4 convergence study in T10).
    Returns (x0, s) with x2(t) = amp * e^{sigma t} cos(w t).
    """
    roots = qep_damped_roots(p)
    upper = [r for r in roots if r.imag > 0.0]
    s = min(upper, key=lambda r: r.imag)  # body mode = lower frequency
    phi1 = ((s * s + (p.gamma_s + p.gamma_sky) * s + p.omega_s_sq)
            / (p.gamma_s * s + p.omega_s_sq))
    x0 = (amp * phi1.real, amp, amp * (s * phi1).real, amp * s.real)
    return x0, s


def mechanical_energy(state: np.ndarray, p) -> np.ndarray:
    """Total mechanical energy of the passive system at zero road."""
    x1, x2, v1, v2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    return (0.5 * v2 * v2 + 0.5 * p.mu * v1 * v1
            + 0.5 * p.omega_s_sq * (x2 - x1) ** 2
            + 0.5 * p.mu * p.omega_t_sq * x1 ** 2)
