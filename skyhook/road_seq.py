"""Host-side discrete road-excitation sequences (plan C, F0).

Plan C decision 5 / section 3 (F0): training needs arbitrary random
excitation series; the analytic road classes (skyhook/road.py) stay
untouched. Generators here produce (z, zdot) on the RK4 HALF-STEP grid
t_i = i*dt/2 -- the exact times a rollout evaluates -- written element-wise
into model3's sequence fields; the kernel does the linear interpolation
(road_eval_seq) and NEVER differences the discrete series (plan A contract
rule 1). Everything here is plain numpy on the host and gradient-detached
by construction (domain-randomization principle 3).

Two-component decomposition discipline (plan C v2.3, decision 10): every
sequence stores the material-roughness component z_ar (AR(1) background,
roughness switch included) and the discrete-event component z_evt
(bump / pothole / dip half-sines) SEPARATELY. The simulation rolls the
SUM; the mode-2 preview channel exposes ONLY z_evt (the forward sensor
sees geometric events, not material roughness).

P2 alignment convention: AR series are burned in (transient discarded),
then shifted so z_r(0) = 0, and zdot_r(0) is set to 0 -- a quasi-static
start compatible with the default rest IC (nonzero-road discipline,
D4 sqrt(0) caveat; the MS reduction has no sqrt, the convention is kept
anyway so every family starts the same way).

Seed determinism: every stochastic generator takes an integer seed (or
numpy Generator); same seed -> byte-identical arrays (domain-randomization
principle 1, docs/parameters_reference.md section 4.5).

Negative-amplitude half-sines (potholes / dips) are supported HERE
directly; skyhook/road.py's analytic BumpRoad keeps its amp >= 0
constraint (T19 cross-checks negative pulses against the closed-form
formula, not against the analytic class).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

# --- AR(1) roughness family defaults (parameters_reference.md section 4.5) ---
AR_PHI = 0.98          # per-half-step correlation (tau ~ 25 ms at dt = 1 ms)
ROUGHNESS_LEVELS = {"fine": 0.003, "asphalt": 0.006, "concrete": 0.012}

# --- discrete-event family defaults (plan C decision 10 / v2.1) --------------
EVT_WIDTH_MIN = 0.05   # s; >= 10 preview pooling points at 5 ms (v2.1)
EVT_GAP_MIN = 0.15     # s; minimum spacing between consecutive events
EVT_N_MAX = 3          # events per episode (suggested band 1..3)
EVT_T_MARGIN = 0.10    # s; keep events clear of the episode edges

# event type parameter bands (amp in m, width in s; sign encodes the type)
_EVENT_SPECS = {
    "bump":    {"amp": (0.010, 0.030), "width": (0.05, 0.10)},
    "pothole": {"amp": (-0.030, -0.015), "width": (0.05, 0.08)},  # short, deep
    "dip":     {"amp": (-0.015, -0.008), "width": (0.10, 0.20)},  # wide, shallow
}


@dataclass
class Event:
    """One signed half-sine road event on [t0, t0+width].

    amp > 0: bump; amp < 0: pothole (short) or dip (wide). The `kind`
    label is diagnostics only -- the shape is the same half-sine for all
    kinds; only the (amp, width) band differs (sample_events).
    """

    amp: float
    t0: float
    width: float
    kind: str = "generic"

    def __post_init__(self):
        if not np.isfinite(self.amp):
            raise ValueError(f"amp must be finite, got {self.amp}")
        if self.width <= 0.0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.t0 < 0.0:
            raise ValueError(f"t0 must be >= 0, got {self.t0}")

    def eval(self, t):
        """(z, zdot) of the half-sine pulse at times t (closed form)."""
        t = np.asarray(t, dtype=float)
        ph = np.pi * (t - self.t0) / self.width
        inside = (t >= self.t0) & (t <= self.t0 + self.width)
        ph_safe = np.clip(ph, 0.0, np.pi)
        z = np.where(inside, self.amp * np.sin(ph_safe), 0.0)
        zd = np.where(inside, self.amp * (np.pi / self.width) * np.cos(ph_safe), 0.0)
        return z, zd


@dataclass
class RoadSeq:
    """Discrete excitation on the half-step grid t_i = i*dt/2, i = 0..2n.

    z/zd are the TOTAL profile the simulation rolls; z_ar/z_evt are the
    component split the preview channel needs (z == z_ar + z_evt exactly;
    zd_evt is the event-component velocity the mode-2 preview window
    reads, plan C decision 9 M2-C -- F2 increment). meta records the
    family parameters (diagnostics / tests).
    """

    dt: float
    n_steps: int
    z: np.ndarray
    zd: np.ndarray
    z_ar: np.ndarray
    z_evt: np.ndarray
    zd_evt: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        m = 2 * self.n_steps + 1
        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        for name in ("z", "zd", "z_ar", "z_evt", "zd_evt"):
            arr = getattr(self, name)
            if arr.shape != (m,):
                raise ValueError(f"{name} must have shape ({m},), got {arr.shape}")

    @property
    def m(self) -> int:
        """Number of half-step grid points."""
        return 2 * self.n_steps + 1

    @property
    def t_grid(self) -> np.ndarray:
        return np.arange(self.m, dtype=np.float64) * (self.dt / 2.0)


def sample_analytic(road, n_steps: int, dt: float) -> RoadSeq:
    """Sample an analytic road (road.py) on the half-step grid.

    Cross-check family (T19): analytic derivatives are stored exactly, so
    a model3 rollout on the sampled series reproduces the model2b analytic
    rollout up to sin/cos ulp differences. Component split: the whole
    profile counts as material (z_ar = z, z_evt = 0) -- analytic sampling
    is a test/baseline family, not a preview-semantics carrier.
    """
    tg = np.arange(2 * n_steps + 1, dtype=np.float64) * (dt / 2.0)
    z, zd = road.eval(tg)
    z = np.ascontiguousarray(z, dtype=np.float64)
    zd = np.ascontiguousarray(zd, dtype=np.float64)
    return RoadSeq(dt=dt, n_steps=n_steps, z=z, zd=zd,
                   z_ar=z.copy(), z_evt=np.zeros_like(z),
                   zd_evt=np.zeros_like(z),
                   meta={"family": "analytic", "road": repr(road)})


def _burn_len(phi: float) -> int:
    """Burn-in samples so the zero-state transient decays below ~1e-4."""
    if phi <= 0.0:
        return 1
    return max(64, int(np.ceil(np.log(1.0e-4) / np.log(phi))))


def _ar1_component(sigma_profile: np.ndarray, phi: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) on the half-step grid with a per-point target std.

    z[j] = phi*z[j-1] + sqrt(1-phi^2)*sigma[j]*w[j] has stationary std
    sigma under constant sigma; a step in sigma (roughness switch) lets
    the variance relax to the new level over ~1/(1-phi) half-steps --
    the plan C decision-10 sigma-step semantics. Burn-in transient is
    discarded, then the whole series is shifted so z[0] = 0 (P2).
    """
    m = sigma_profile.size
    burn = _burn_len(phi)
    sig_ext = np.concatenate([np.full(burn, sigma_profile[0]), sigma_profile])
    e = np.sqrt(1.0 - phi * phi) * sig_ext * rng.standard_normal(burn + m)
    z_full = lfilter(np.array([1.0]), np.array([1.0, -phi]), e)
    z = np.ascontiguousarray(z_full[burn:], dtype=np.float64)
    return z - z[0]


def _central_diff(z: np.ndarray, h: float) -> np.ndarray:
    """zdot on the half-step grid: central differences, P2 endpoints.

    The kernel interpolates; it never differences (contract rule 1), so
    the derivative is computed HERE. zd[0] = 0 is the P2 alignment
    convention; the far end uses a one-sided difference (read only by the
    last stage-4 evaluation).
    """
    zd = np.empty_like(z)
    zd[1:-1] = (z[2:] - z[:-2]) / (2.0 * h)
    zd[0] = 0.0
    zd[-1] = (z[-1] - z[-2]) / h
    return zd


def ar_switching(sigma1: float, sigma2: float, t_switch: float,
                 n_steps: int, dt: float, phi: float = AR_PHI,
                 seed=None) -> RoadSeq:
    """Roughness-switching AR(1) family (plan C mode 1 scenario).

    sigma steps from sigma1 to sigma2 at t_switch (one step per episode,
    domain-randomization axis 3). No events: z_evt is identically zero.
    """
    if sigma1 <= 0.0 or sigma2 <= 0.0:
        raise ValueError("roughness levels must be > 0")
    t_end = n_steps * dt
    if not (0.0 <= t_switch <= t_end):
        raise ValueError(f"t_switch={t_switch} outside (0, {t_end})")
    rng = np.random.default_rng(seed)
    tg = np.arange(2 * n_steps + 1, dtype=np.float64) * (dt / 2.0)
    sigma = np.where(tg < t_switch, sigma1, sigma2)
    z = _ar1_component(sigma, phi, rng)
    zd = _central_diff(z, dt / 2.0)
    meta = {"family": "ar_switch", "sigma1": sigma1, "sigma2": sigma2,
            "t_switch": t_switch, "phi": phi, "seed": seed}
    return RoadSeq(dt=dt, n_steps=n_steps, z=z, zd=zd,
                   z_ar=z.copy(), z_evt=np.zeros_like(z),
                   zd_evt=np.zeros_like(z), meta=meta)


def events_ar(events, sigma1: float, sigma2: float, t_switch,
              n_steps: int, dt: float, phi: float = AR_PHI,
              seed=None) -> RoadSeq:
    """AR background (with optional roughness switch) + discrete events.

    The mode-2 scenario family (plan C decision 10): z = z_ar + z_evt with
    z_evt the signed half-sine sum (bumps / potholes / dips, width
    >= EVT_WIDTH_MIN when built via sample_events) and z_ar the switching
    AR background. t_switch=None keeps a constant sigma1 background.
    Velocity = analytic event derivative + central-difference AR
    derivative (exact for the pulses, second-order for the background).
    """
    events = list(events)
    if not events:
        raise ValueError("events must be non-empty (use ar_switching for a "
                         "pure AR road)")
    t_end = n_steps * dt
    if t_switch is not None and not (0.0 <= t_switch <= t_end):
        raise ValueError(f"t_switch={t_switch} outside (0, {t_end})")
    rng = np.random.default_rng(seed)
    tg = np.arange(2 * n_steps + 1, dtype=np.float64) * (dt / 2.0)

    # AR background (constant sigma when t_switch is None)
    sigma = np.full(tg.size, sigma1)
    if t_switch is not None:
        sigma = np.where(tg < t_switch, sigma1, sigma2)
    z_ar = _ar1_component(sigma, phi, rng)
    zd_ar = _central_diff(z_ar, dt / 2.0)

    # event component (analytic half-sine pulses, negative amps allowed)
    z_evt = np.zeros_like(tg)
    zd_evt = np.zeros_like(tg)
    for ev in events:
        ze, zde = ev.eval(tg)
        z_evt += ze
        zd_evt += zde

    z = z_ar + z_evt
    zd = zd_ar + zd_evt
    meta = {"family": "events_ar", "sigma1": sigma1, "sigma2": sigma2,
            "t_switch": t_switch, "phi": phi, "seed": seed,
            "events": [(e.kind, e.amp, e.t0, e.width) for e in events]}
    return RoadSeq(dt=dt, n_steps=n_steps, z=z, zd=zd,
                   z_ar=z_ar, z_evt=z_evt, zd_evt=zd_evt, meta=meta)


def sample_events(rng: np.random.Generator, n_steps: int, dt: float,
                  t_margin: float = EVT_T_MARGIN,
                  gap_min: float = EVT_GAP_MIN) -> list:
    """Random event list (domain-randomization axis 4, suggested band).

    1..EVT_N_MAX signed events, widths >= EVT_WIDTH_MIN, consecutive
    event windows separated by >= gap_min, kept t_margin clear of the
    episode edges. Type drawn from {bump, pothole, dip} with the
    _EVENT_SPECS amp/width bands.
    """
    t_end = n_steps * dt
    if t_end < 2.0 * t_margin + EVT_WIDTH_MIN + gap_min:
        raise ValueError(f"episode too short ({t_end}s) for event sampling")
    n_events = int(rng.integers(1, EVT_N_MAX + 1))
    events = []
    cursor = t_margin
    for _ in range(n_events):
        kind = str(rng.choice(list(_EVENT_SPECS)))
        a_lo, a_hi = _EVENT_SPECS[kind]["amp"]
        w_lo, w_hi = _EVENT_SPECS[kind]["width"]
        amp = float(rng.uniform(a_lo, a_hi))
        width = float(rng.uniform(w_lo, w_hi))
        t_hi = t_end - t_margin - width
        if t_hi < cursor:
            break  # no room left for another event
        t0 = float(rng.uniform(cursor, t_hi))
        events.append(Event(amp=amp, t0=t0, width=width, kind=kind))
        cursor = t0 + width + gap_min
    return events
