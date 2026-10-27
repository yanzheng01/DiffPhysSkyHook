"""Shared D4 batch fixture: B=8 mixed params / roads / ICs.

Single source for tests/test_d4_batch.py (CPU anchors) and the GPU worker
(tests/gpu_worker.py + tests/test_d4_gpu.py), so the CPU and CUDA runs
exercise byte-identical inputs.

Numerical caveat (D4 probe finding): flat-road elements get a nonzero IC
(free decay) -- J = RMS(acc) = 0 would make dual(j_val_b) = sqrt'(0) =
0/0 = nan and poison j_mean's dual chain.
"""

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
