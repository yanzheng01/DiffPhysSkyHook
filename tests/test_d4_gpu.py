"""D4 GPU parity tests (T11 CPU-GPU leg, T13 GPU leg).

tests/gpu_worker.py runs the shared D4 batch fixture on CUDA in a
subprocess (Quadrants binds the backend once per process, so the CPU
reference and the CUDA run cannot share one); it saves an .npz that this
test compares against the CPU reference:

  T11 GPU leg: state / a2 / fd / mask / ft / j / j_mean / final_state /
      grad_ad element-wise |CPU - GPU| < 1e-12 (observed on the dev box:
      bit-exact, diff = 0.0).
  T13 GPU leg: the worker's in-process three-way per-element gradients
      (reverse Tape vs FwdMode JVP vs central FD, h = 1e-4, active-set
      stability asserted in the worker printout) with rel < 1e-5.

Plan discipline: run pytest for this file OUTSIDE the sandbox (CUDA needs
real device access; sandboxed runs silently fall back to / fail on the
driver).
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import quadrants as qd

from d4_fixture import B, DT, N_STEPS, PARAMS, ROADS, X0S
from skyhook import model2b
from skyhook.sim2b import _setup, run_j_chain, run_semi_batch

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "gpu_worker.py")
TOL = 1.0e-12


@pytest.fixture(scope="module")
def gpu_out(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("d4_gpu") / "out.npz")
    r = subprocess.run([sys.executable, WORKER, out],
                       capture_output=True, text=True, timeout=1200)
    print(r.stdout)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "arch=cuda" in r.stdout, "worker did not start on CUDA"
    return np.load(out)


class TestT11GpuParity:
    def test_channels_match_cpu(self, gpu_out):
        cpu = run_semi_batch(PARAMS, ROADS, x0s=X0S,
                             t_end=N_STEPS * DT, dt=DT)
        _setup(PARAMS, ROADS, X0S)
        with qd.ad.Tape(loss=model2b.j_mean):
            run_j_chain(N_STEPS, DT, B)
        grad_cpu = model2b.pg_b.grad.to_numpy()[:B]

        checks = [
            ("state", gpu_out["state"], cpu.state),
            ("a2", gpu_out["a2"], cpu.a2),
            ("fd", gpu_out["fd"], cpu.fd),
            ("mask", gpu_out["mask"], cpu.mask),
            ("ft", gpu_out["ft"], cpu.ft),
            ("final_state", gpu_out["final_state"], cpu.final_state),
            ("j", gpu_out["j"], cpu.j),
            ("grad_ad", gpu_out["grad_ad"], grad_cpu),
        ]
        for name, g, c in checks:
            diff = float(np.max(np.abs(np.asarray(g) - np.asarray(c))))
            print(f"[D4] T11 GPU {name}: max |cpu-gpu| = {diff:.3e}")
            assert diff < TOL
        assert abs(float(gpu_out["j_mean"]) - cpu.j_mean) < TOL

    def test_three_way_gradients_on_gpu(self, gpu_out):
        """T13 GPU leg: in-worker three-way agreement, rel < 1e-5."""
        idx = gpu_out["jvp_indices"]
        ad = gpu_out["grad_ad"][idx]
        jvp = gpu_out["grad_jvp"]
        fd = gpu_out["grad_fd"]
        rel = lambda a, b: np.abs(a - b) / np.maximum(np.abs(b), 1e-300)  # noqa: E731
        print(f"[D4] T13 GPU: AD={ad}, JVP={jvp}, FD={fd}")
        print(f"     rel AD/FD={rel(ad, fd)}, rel JVP/FD={rel(jvp, fd)}, "
              f"rel AD/JVP={rel(ad, jvp)}")
        print(f"     active-set stable={gpu_out['stable']}")
        assert np.all(gpu_out["stable"])
        assert np.all(np.isfinite(ad)) and np.all(np.isfinite(jvp))
        assert np.all(rel(ad, fd) < 1.0e-5)
        assert np.all(rel(jvp, fd) < 1.0e-5)
        assert np.all(rel(ad, jvp) < 1.0e-5)
