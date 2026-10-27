"""Analytic road profiles: Road.eval(t) -> (z_r, zdot_r) in closed form.

Contract (docs/plan_a_dev_test_plan.md section 3, rule 1): roads must be
evaluable at arbitrary t (including RK4 half steps) with an analytic
derivative; discrete spectra are handled later via interpolation.

The same formulas are implemented kernel-side in skyhook/model.py
(road_eval); tests/test_d1_passive.py cross-checks the two implementations.

Ramp-in (rule 2): sine roads fade in over `ramp_t` seconds with a smoothstep
envelope (C1), so a quiescent initial state is not shocked at t = 0.
"""

import numpy as np

ROAD_ZERO = 0
ROAD_SINE = 1
ROAD_BUMP = 2


def _smoothstep(t, ramp_t):
    """Smoothstep envelope e(t), de/dt on [0, ramp_t]; e = 1 afterwards."""
    t = np.asarray(t, dtype=float)
    if ramp_t <= 0.0:
        e = np.ones_like(t)
        de = np.zeros_like(t)
        return e, de
    s = np.clip(t / ramp_t, 0.0, 1.0)
    e = 3.0 * s * s - 2.0 * s * s * s
    de = np.where((t > 0.0) & (t < ramp_t), (6.0 * s - 6.0 * s * s) / ramp_t, 0.0)
    return e, de


class ZeroRoad:
    """Flat road (free-vibration / decay tests)."""

    def eval(self, t):
        t = np.asarray(t, dtype=float)
        z = np.zeros_like(t)
        zd = np.zeros_like(t)
        return z, zd

    def kernel_args(self):
        """(kind, amp, w, t0, width, ramp_t) with safe defaults for unused slots."""
        return (ROAD_ZERO, 0.0, 1.0, 0.0, 1.0, 0.0)


class SineRoad:
    """Sinusoidal road z = amp * sin(2*pi*f*t) * e(t) with smoothstep ramp-in."""

    def __init__(self, amp: float, freq_hz: float, ramp_t: float = 0.5):
        if amp < 0.0:
            raise ValueError("amp must be >= 0")
        if freq_hz <= 0.0:
            raise ValueError("freq_hz must be > 0")
        if ramp_t < 0.0:
            raise ValueError("ramp_t must be >= 0")
        self.amp = float(amp)
        self.freq_hz = float(freq_hz)
        self.ramp_t = float(ramp_t)

    def eval(self, t):
        t = np.asarray(t, dtype=float)
        w = 2.0 * np.pi * self.freq_hz
        e, de = _smoothstep(t, self.ramp_t)
        z = self.amp * np.sin(w * t) * e
        zd = self.amp * (w * np.cos(w * t) * e + np.sin(w * t) * de)
        return z, zd

    def kernel_args(self):
        return (ROAD_SINE, self.amp, 2.0 * np.pi * self.freq_hz, 0.0, 1.0, self.ramp_t)


class BumpRoad:
    """Half-sine bump: z = amp*sin(pi*(t-t0)/width) on [t0, t0+width], else 0."""

    def __init__(self, amp: float, t0: float, width: float):
        if amp < 0.0:
            raise ValueError("amp must be >= 0")
        if width <= 0.0:
            raise ValueError("width must be > 0")
        self.amp = float(amp)
        self.t0 = float(t0)
        self.width = float(width)

    def eval(self, t):
        t = np.asarray(t, dtype=float)
        w = self.width
        ph = np.pi * (t - self.t0) / w
        inside = (t >= self.t0) & (t <= self.t0 + w)
        ph_safe = np.clip(ph, 0.0, np.pi)
        z = np.where(inside, self.amp * np.sin(ph_safe), 0.0)
        zd = np.where(inside, self.amp * (np.pi / w) * np.cos(ph_safe), 0.0)
        return z, zd

    def kernel_args(self):
        return (ROAD_BUMP, self.amp, 1.0, self.t0, self.width, 0.0)
