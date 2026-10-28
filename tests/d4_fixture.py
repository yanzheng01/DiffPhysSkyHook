"""Shared D4 batch fixture: B=8 mixed params / roads / ICs.

Single source for tests/test_d4_batch.py (CPU anchors) and the GPU worker
(tests/gpu_worker.py + tests/test_d4_gpu.py), so the CPU and CUDA runs
exercise byte-identical inputs.

Numerical caveat (D4 probe finding): flat-road elements get a nonzero IC
(free decay) -- J = RMS(acc) = 0 would make dual(j_val_b) = sqrt'(0) =
0/0 = nan and poison j_mean's dual chain.

Plan B (E4) adds the shared command block FCMDS (B, N_STEPS): deterministic
uniform commands on the N-force scale of the (c_min, c_max) window, with
adversarial spikes on one element so every batch run mixes in-band and
out-of-band steps. Same PARAMS / ROADS / X0S as the D4 fixture, so the
flat-road elements keep their nonzero ICs (J > 0 dual caveat above).
"""

import numpy as np

from skyhook.params import SemiActiveParams
from skyhook.road import BumpRoad, SineRoad, ZeroRoad

DT = 1.0e-3
N_STEPS = 1000
B = 8

_P0 = SemiActiveParams()
PARAMS = [SemiActiveParams(gamma_sky=_P0.gamma_sky * (0.7 + 0.15 * i),
                            c_min=_P0.c_min * (0.8 + 0.1 * (i % 3)),
                            c_max=_P0.c_max * (0.8 + 0.1 * (i % 4)))
          for i in range(B)]
ROADS = [BumpRoad(amp=0.05, t0=0.05, width=0.1),
         SineRoad(amp=0.02, freq_hz=1.2, ramp_t=0.5),
         ZeroRoad(),
         BumpRoad(amp=0.08, t0=0.1, width=0.2),
         SineRoad(amp=0.03, freq_hz=8.0, ramp_t=0.5),
         BumpRoad(amp=0.02, t0=0.2, width=0.05),
         ZeroRoad(),
         SineRoad(amp=0.01, freq_hz=2.5, ramp_t=0.5)]
X0S = [(0.0, 0.0, 0.0, 0.0)] * B
X0S[2] = (0.0, 0.01, 0.0, 0.0)          # flat road -> nonzero IC (J > 0)
X0S[6] = (0.005, -0.01, 0.02, -0.03)


def perturbed(params_list, b: int, delta: float):
    """Copy of params_list with gamma_sky[b] shifted by delta."""
    out = list(params_list)
    out[b] = SemiActiveParams(gamma_sky=params_list[b].gamma_sky + delta,
                              c_min=params_list[b].c_min,
                              c_max=params_list[b].c_max)
    return out


# --- plan B (E4) shared force-command block -----------------------------------
_FCMD_RNG = np.random.default_rng(42)
FCMDS = _FCMD_RNG.uniform(-1.0, 1.0, size=(B, N_STEPS))
FCMDS[3, ::10] *= 8.0  # adversarial spikes on b=3 -> out-of-band samples


def perturbed_cmd(f_cmds, b: int, n: int, delta: float):
    """Copy of the command block with f_cmds[b, n] shifted by delta."""
    out = np.array(f_cmds, dtype=np.float64, copy=True)
    out[b, n] += delta
    return out


class _Seed2D(np.ndarray):
    """FwdMode seed adapter for 2-D fields (plan B E4, fc_b JVP probes).

    The FwdMode seed API expects a flat list whose len() equals the field's
    element count AND a numpy payload whose shape matches the field shape --
    a flat list of 262144 scalars cannot satisfy both (np.array stays 1-D).
    This ndarray subclass bridges the two: __bool__ -> True selects the
    "seed set" branch, __len__ -> size satisfies the flat-length assert, and
    np.array(seed, float32) keeps the (B_MAX, T_B_MAX) shape for from_numpy.
    """

    def __bool__(self):
        return True

    def __len__(self):
        return self.size


def fwd_seed_2d(b: int, n: int):
    """Unit JVP seed e_(b, n) for the (B_MAX, T_B_MAX) command field."""
    from skyhook import model2b  # local: keep the fixture module import-light

    seed = np.zeros((model2b.B_MAX, model2b.T_B_MAX), dtype=np.float64)
    seed[b, n] = 1.0
    return seed.view(_Seed2D)
