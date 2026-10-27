"""D4 GPU worker: batched model II on CUDA (run as a script, not via pytest).

Quadrants binds the backend once at qd.init; this script calls
ensure_init(arch=qd.cuda) BEFORE importing any skyhook model module so all
model fields are placed on the CUDA runtime. It then runs the shared D4
batch fixture (tests/d4_fixture.py):

  - forward batch rollout  -> state / a2 / fd / mask / j / j_mean
  - reverse Tape on j_mean -> full dJ_mean/dgamma_sky gradient vector
  - forward JVP (seed e_i) and central FD (h = 1e-4) for i in {0, 3, 6}

Everything is saved to an .npz (argv[1]) for the CPU-side parity test
(tests/test_d4_gpu.py, T11 CPU-GPU leg; tolerance 1e-12 element-wise) and
the in-worker three-way summary (T13 GPU leg) is printed to stdout.

Plan discipline: launch OUTSIDE the sandbox (CUDA needs real device
access; the pytest wrapper that spawns this worker must run with
dangerouslyDisableSandbox to avoid a silent CPU fallback).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import quadrants as qd  # noqa: E402

from skyhook._runtime import ensure_init  # noqa: E402

ensure_init(arch=qd.cuda)  # bind CUDA BEFORE importing model modules

from skyhook import model2b  # noqa: E402
from skyhook.sim2b import _setup, run_j_chain, run_semi_batch  # noqa: E402

from d4_fixture import B, DT, N_STEPS, PARAMS, ROADS, X0S, perturbed  # noqa: E402

JVP_INDICES = (0, 3, 6)
H = 1.0e-4


def main(out_path: str):
    # forward batch (also the FD reference run)
    res0 = run_semi_batch(PARAMS, ROADS, x0s=X0S, t_end=N_STEPS * DT, dt=DT)

    # reverse: full gradient vector from one Tape
    _setup(PARAMS, ROADS, X0S)
    with qd.ad.Tape(loss=model2b.j_mean):
        run_j_chain(N_STEPS, DT, B)
    grad_ad = model2b.pg_b.grad.to_numpy()[:B]

    # forward JVP + central FD per probe element
    grad_jvp = {}
    grad_fd = {}
    stable = {}
    for i in JVP_INDICES:
        seed = [0.0] * model2b.B_MAX  # seed spans the FULL field shape
        seed[i] = 1.0
        _setup(PARAMS, ROADS, X0S)
        with qd.ad.FwdMode(loss=model2b.j_mean, param=model2b.pg_b,
                           seed=seed):
            run_j_chain(N_STEPS, DT, B)
        grad_jvp[i] = model2b.j_mean.dual[None]

        up = run_semi_batch(perturbed(PARAMS, i, +H), ROADS, x0s=X0S,
                            t_end=N_STEPS * DT, dt=DT)
        dn = run_semi_batch(perturbed(PARAMS, i, -H), ROADS, x0s=X0S,
                            t_end=N_STEPS * DT, dt=DT)
        grad_fd[i] = (up.j_mean - dn.j_mean) / (2.0 * H)
        stable[i] = (np.array_equal(res0.mask, up.mask)
                     and np.array_equal(res0.mask, dn.mask))

    np.savez(out_path,
             state=res0.state, a2=res0.a2, fd=res0.fd, mask=res0.mask,
             ft=res0.ft, j=res0.j, j_mean=res0.j_mean,
             final_state=res0.final_state, grad_ad=grad_ad,
             jvp_indices=np.array(JVP_INDICES),
             grad_jvp=np.array([grad_jvp[i] for i in JVP_INDICES]),
             grad_fd=np.array([grad_fd[i] for i in JVP_INDICES]),
             stable=np.array([stable[i] for i in JVP_INDICES]))

    print(f"[GPU] j_mean={res0.j_mean:.12e}")
    print(f"[GPU] grad_ad={np.round(grad_ad, 8)}")
    for i in JVP_INDICES:
        rel = abs(grad_ad[i] - grad_fd[i]) / max(abs(grad_fd[i]), 1e-300)
        print(f"[GPU] i={i}: AD={grad_ad[i]:.10e} JVP={grad_jvp[i]:.10e} "
              f"FD={grad_fd[i]:.10e} rel_ad_fd={rel:.3e} stable={stable[i]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/d4_gpu_out.npz")
