"""Parameter container for the plan-A quarter-car model (m2 == 1 gauge).

Gains follow docs/plan_a_skyhook_model.md section 2:
  mu       = m1/m2 (un sprung / sprung mass ratio, gauge-fixed)
  omega_t^2 = k1/mu (tire stiffness gain)
  omega_s^2 = k2     (suspension stiffness gain, m2 == 1)
  gamma_t  = c1/mu   (tire damping gain)
  gamma_s  = c2      (suspension damping gain)

Positivity/ordinal constraints are enforced here (interface contract rule 6);
the unconstrained (exp-domain) view is provided for the D5 optimizer path.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PassiveParams:
    """Quarter-car gains; gamma_sky = 0 gives the passive baseline (D1)."""

    mu: float = 0.1
    omega_t_sq: float = (2.0 * math.pi * 12.0) ** 2  # wheel-hop ~12 Hz
    omega_s_sq: float = (2.0 * math.pi * 1.5) ** 2   # body mode ~1.5 Hz
    gamma_t: float = 2.0 * 0.02 * (2.0 * math.pi * 12.0)  # tire zeta ~ 0.02
    gamma_s: float = 2.0 * 0.10 * (2.0 * math.pi * 1.5)   # susp. zeta ~ 0.10
    gamma_sky: float = 0.0                            # ideal skyhook gain (model I)

    def __post_init__(self):
        if not (self.mu > 0.0):
            raise ValueError(f"mu must be > 0, got {self.mu}")
        if not (self.omega_t_sq > 0.0 and self.omega_s_sq > 0.0):
            raise ValueError("stiffness gains must be > 0")
        if self.gamma_t < 0.0 or self.gamma_s < 0.0 or self.gamma_sky < 0.0:
            raise ValueError("damping gains must be >= 0")
        for name in ("mu", "omega_t_sq", "omega_s_sq", "gamma_t", "gamma_s", "gamma_sky"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")

    @property
    def omega_t(self) -> float:
        return math.sqrt(self.omega_t_sq)

    @property
    def omega_s(self) -> float:
        return math.sqrt(self.omega_s_sq)

    @property
    def zeta_s(self) -> float:
        """Body-mode damping-ratio anchor zeta = gamma_s / (2 omega_s)."""
        return self.gamma_s / (2.0 * self.omega_s)

    @property
    def zeta_eff(self) -> float:
        """Model-I body-mode damping anchor (gamma_s + gamma_sky)/(2 omega_s)."""
        return (self.gamma_s + self.gamma_sky) / (2.0 * self.omega_s)

    @classmethod
    def from_unconstrained(cls, raw_mu, raw_omega_t_sq, raw_omega_s_sq, raw_gamma_t, raw_gamma_s):
        """Map unconstrained reals to the positive parameter domain via exp.

        Used by the optimizer (D5); the interface never exposes raw values.
        """
        return cls(
            mu=math.exp(raw_mu),
            omega_t_sq=math.exp(raw_omega_t_sq),
            omega_s_sq=math.exp(raw_omega_s_sq),
            gamma_t=math.exp(raw_gamma_t),
            gamma_s=math.exp(raw_gamma_s),
        )


@dataclass(frozen=True)
class SemiActiveParams:
    """Model-II gains (docs/plan_a_skyhook_model.md section 4).

    The paired suspension damper is the projected force
      F_d = clip(gamma_sky * v2, min(c_min, c_max)*v_rel, max(c_min, c_max)*v_rel)
    so the passive gamma_s of model I is replaced by the (c_min, c_max) pair;
    gamma_sky only enters as the ideal-skyhook force command. c_min == c_max
    is the fixed-gain degenerate damper (passive degeneracy, T8); the strict
    c_min < c_max case is the general semi-active domain.
    """

    mu: float = 0.1
    omega_t_sq: float = (2.0 * math.pi * 12.0) ** 2
    omega_s_sq: float = (2.0 * math.pi * 1.5) ** 2
    gamma_t: float = 2.0 * 0.02 * (2.0 * math.pi * 12.0)
    gamma_sky: float = 2.0 * 0.30 * (2.0 * math.pi * 1.5)  # zeta_sky = 0.30
    c_min: float = 2.0 * 0.03 * (2.0 * math.pi * 1.5)      # zeta ~ 0.03
    c_max: float = 2.0 * 0.30 * (2.0 * math.pi * 1.5)     # zeta ~ 0.30

    def __post_init__(self):
        if not (self.mu > 0.0):
            raise ValueError(f"mu must be > 0, got {self.mu}")
        if not (self.omega_t_sq > 0.0 and self.omega_s_sq > 0.0):
            raise ValueError("stiffness gains must be > 0")
        if self.gamma_t < 0.0 or self.gamma_sky < 0.0:
            raise ValueError("gamma_t / gamma_sky must be >= 0")
        if not (0.0 <= self.c_min <= self.c_max):
            raise ValueError(f"need 0 <= c_min <= c_max, got ({self.c_min}, {self.c_max})")
        for name in ("mu", "omega_t_sq", "omega_s_sq", "gamma_t", "gamma_sky",
                     "c_min", "c_max"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")

    @property
    def omega_s(self) -> float:
        return math.sqrt(self.omega_s_sq)

    @classmethod
    def from_unconstrained(cls, raw_gamma_sky, raw_c_min, raw_c_max_shift,
                           mu=0.1, omega_t_sq=(2.0 * math.pi * 12.0) ** 2,
                           omega_s_sq=(2.0 * math.pi * 1.5) ** 2,
                           gamma_t=2.0 * 0.02 * (2.0 * math.pi * 12.0)):
        """Unconstrained -> constrained domain via softplus (contract rule 6).

        gamma_sky >= 0 and c_min >= 0 via softplus; the ordering c_max > c_min
        via c_max = c_min + softplus(shift). Hardware-given gains stay fixed.
        """
        sp = lambda x: math.log1p(math.exp(x))  # noqa: E731  (softplus)
        c_min = sp(raw_c_min)
        return cls(mu=mu, omega_t_sq=omega_t_sq, omega_s_sq=omega_s_sq,
                   gamma_t=gamma_t, gamma_sky=sp(raw_gamma_sky),
                   c_min=c_min, c_max=c_min + sp(raw_c_max_shift))
