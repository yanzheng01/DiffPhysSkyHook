"""E4 batch + GPU tests for the command-driven channel (plan B, Gate T18).

tests/test_e4_force_batch.py extends the D4 batch battery to the external
force-command block FCMDS (tests/d4_fixture.py):

  T18 batch-vs-expand: run_semi_cmd_batch (B = 8) vs B independent
      run_semi_cmd rollouts, element-wise on state / a2 / fd / mask / ft /
      final_state. Expected (and observed) bit-exact: the per-element step
      body is identical arithmetic (T11 pattern). res.mask is the COMMAND
      out-of-band rate on this rollout.
  T18 batch adjoint structure (E2 semantics at batch scale, one reverse
      Tape -> the whole (B, n_steps) adjoint dJ_mean/dfc_b):
        - untouched slots (n >= n_steps or b >= B) carry exactly 0;
        - every in-band step carries a nonzero adjoint (direct channel
          dacc[n]/dfc = -1, generic position);
        - a step whose RECORDED mask is 1 may still carry a nonzero
          adjoint: the record window is the stage-1 (whole-step) state,
          the stage-2..4 RK4 states can be in-band. Empirically 41/6441
          clipped steps on the shared fixture.
  T18 three-way semantics: reverse / FwdMode JVP (2-D seed adapter,
      tests.d4_fixture.fwd_seed_2d) / central FD (h = 1e-4, mask stability
      asserted) at three runtime-picked slots -- in-band, mixed-clip, and
      fully saturated. The fully saturated slot has adjoint == JVP == FD ==
      0.0 exactly: the command never enters any stage window, the whole
      trajectory is invariant under +/-h (bit-identical J).
  T18 CPU-GPU parity + GPU in-worker three-way: the shared worker
      (tests/gpu_worker.py) runs the same fixture on CUDA and saves an
      .npz; tolerance 1e-12 element-wise, three-way rel < 1e-5.

Plan discipline: run pytest for the GPU class OUTSIDE the sandbox (CUDA
needs real device access; sandboxed runs silently fall back / fail).
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import quadrants as qd

from d4_fixture import (B, DT, FCMDS, N_STEPS, PARAMS, ROADS, X0S,  # noqa: E402
                        fwd_seed_2d, perturbed_cmd)
from skyhook import model2b
from skyhook.sim2 import run_semi_cmd
from skyhook.sim2b import (
    _setup, _setup_fcmd, run_cmd_j_chain, run_semi_cmd_batch)

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "gpu_worker.py")
TOL = 1.0e-12
H = 1.0e-4  # D3 finding: h=1e-3 straddles stage-level clip switches
T_END = N_STEPS * DT


def _pick_slot(pred):
    for b in range(B):
        for n in range(N_STEPS):
            if pred(b, n):
                return b, n
    raise RuntimeError("no command slot matches the predicate")


class TestT18BatchCmdExpand:
    def test_batch_equals_expanded_singles(self):
        batch = run_semi_cmd_batch(PARAMS, FCMDS, ROADS, x0s=X0S,
                                   t_end=T_END, dt=DT)
        max_diff = 0.0
        for b in range(B):
            single = run_semi_cmd(PARAMS[b], FCMDS[b], ROADS[b], x0=X0S[b],
                                  t_end=T_END, dt=DT)
            max_diff = max(
                max_diff,
                np.max(np.abs(batch.state[b] - single.state)),
                np.max(np.abs(batch.a2[b] - single.a2)),
                np.max(np.abs(batch.fd[b] - single.fd)),
                np.max(np.abs(batch.mask[b] - single.mask)),
                np.max(np.abs(batch.ft[b] - single.ft)),
                np.max(np.abs(batch.final_state[b] - single.final_state)),
            )
        j_numpy = np.sqrt(np.mean(batch.a2 ** 2, axis=1))
        j_diff = float(np.max(np.abs(batch.j - j_numpy)))
        print(f"[E4] T18 expand: max element-wise diff = {max_diff:.3e}, "
              f"max |j(kernel)-j(numpy)| = {j_diff:.3e}, "
              f"oob rate = {float(np.mean(batch.mask)):.4f}, "
              f"j_mean={batch.j_mean:.12e}")
        assert max_diff < TOL
        assert j_diff < TOL
        assert abs(batch.j_mean / float(np.mean(j_numpy)) - 1.0) < TOL

    @staticmethod
    def _reverse_adjoint():
        _setup(PARAMS, ROADS, X0S)
        _setup_fcmd(FCMDS, N_STEPS, B)
        with qd.ad.Tape(loss=model2b.j_mean):
            run_cmd_j_chain(N_STEPS, DT, B)
        return model2b.fc_b.grad.to_numpy()

    def test_adjoint_structure(self):
        """E2 semantics at batch scale: zeros + in-band nonzero + stats."""
        batch = run_semi_cmd_batch(PARAMS, FCMDS, ROADS, x0s=X0S,
                                   t_end=T_END, dt=DT)
        grad = self._reverse_adjoint()
        g = grad[:B, :N_STEPS]

        # untouched slots carry exactly zero (no trajectory reads them)
        assert np.all(grad[:, N_STEPS:] == 0.0)
        assert np.all(grad[B:, :] == 0.0)

        # in-band steps: direct channel dacc[n]/dfc = -1 -> nonzero adjoint
        inband = batch.mask == 0.0
        assert np.all(g[inband] != 0.0)

        n_mixed = int(np.sum((batch.mask == 1.0) & (g != 0.0)))
        n_sat = int(np.sum((batch.mask == 1.0) & (g == 0.0)))
        print(f"[E4] T18 adjoint: max|g|={float(np.max(np.abs(g))):.6e} "
              f"min={float(np.min(g)):.6e} "
              f"nonzero={int(np.count_nonzero(g))}/{g.size}")
        print(f"     clipped steps: mixed( stage2-4 in-band, nonzero )="
              f"{n_mixed}, fully saturated (exact 0)={n_sat}")
        print(f"[E4] T18 adjoint statistics (plan B section 7 deliverable): "
              f"adjoint = dJ_mean/dfc_b, co-design signal for the (B, T) "
              f"command block")
        assert n_mixed > 0
        assert n_sat > 0

    def test_three_way_semantics(self):
        """Reverse / JVP / FD at in-band, mixed-clip, fully-saturated slots."""
        batch = run_semi_cmd_batch(PARAMS, FCMDS, ROADS, x0s=X0S,
                                   t_end=T_END, dt=DT)
        grad = self._reverse_adjoint()
        g = grad[:B, :N_STEPS]

        slots = [
            ("inband", _pick_slot(lambda b, n: batch.mask[b, n] == 0.0
                                  and g[b, n] != 0.0)),
            ("mixed_clip", _pick_slot(lambda b, n: batch.mask[b, n] == 1.0
                                      and g[b, n] != 0.0)),
            ("saturated", _pick_slot(lambda b, n: batch.mask[b, n] == 1.0
                                     and g[b, n] == 0.0)),
        ]
        rel = lambda x, y: abs(x - y) / max(abs(y), 1e-300)  # noqa: E731
        for name, (b, n) in slots:
            _setup(PARAMS, ROADS, X0S)
            _setup_fcmd(FCMDS, N_STEPS, B)
            with qd.ad.FwdMode(loss=model2b.j_mean, param=model2b.fc_b,
                               seed=fwd_seed_2d(b, n)):
                run_cmd_j_chain(N_STEPS, DT, B)
            jvp = model2b.j_mean.dual[None]

            up = run_semi_cmd_batch(PARAMS, perturbed_cmd(FCMDS, b, n, +H),
                                    ROADS, x0s=X0S, t_end=T_END, dt=DT)
            dn = run_semi_cmd_batch(PARAMS, perturbed_cmd(FCMDS, b, n, -H),
                                    ROADS, x0s=X0S, t_end=T_END, dt=DT)
            fd = (up.j_mean - dn.j_mean) / (2.0 * H)
            stable = (np.array_equal(batch.mask, up.mask)
                      and np.array_equal(batch.mask, dn.mask))
            print(f"[E4] T18 slot {name} (b={b}, n={n}): AD={g[b, n]:.10e} "
                  f"JVP={jvp:.10e} FD={fd:.10e} rel_AD_FD={rel(g[b, n], fd):.3e} "
                  f"rel_AD_JVP={rel(g[b, n], jvp):.3e} stable={stable}")
            assert stable
            if name == "saturated":
                # trajectory invariant under +/-h: all three legs exact zero
                assert g[b, n] == 0.0
                assert jvp == 0.0
                assert fd == 0.0
            else:
                assert rel(g[b, n], fd) < 1.0e-5
                assert rel(g[b, n], jvp) < 1.0e-5


class TestT18GpuParity:
    @pytest.fixture(scope="class")
    def gpu_out(self, tmp_path_factory):
        out = str(tmp_path_factory.mktemp("e4_gpu") / "out.npz")
        r = subprocess.run([sys.executable, WORKER, out],
                           capture_output=True, text=True, timeout=3600)
        print(r.stdout)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "arch=cuda" in r.stdout, "worker did not start on CUDA"
        return np.load(out)

    def test_cmd_channels_match_cpu(self, gpu_out):
        cpu = run_semi_cmd_batch(PARAMS, FCMDS, ROADS, x0s=X0S,
                                 t_end=T_END, dt=DT)
        _setup(PARAMS, ROADS, X0S)
        _setup_fcmd(FCMDS, N_STEPS, B)
        with qd.ad.Tape(loss=model2b.j_mean):
            run_cmd_j_chain(N_STEPS, DT, B)
        grad_cpu = model2b.fc_b.grad.to_numpy()[:B, :N_STEPS]

        checks = [
            ("cmd_state", gpu_out["cmd_state"], cpu.state),
            ("cmd_a2", gpu_out["cmd_a2"], cpu.a2),
            ("cmd_force", gpu_out["cmd_force"], cpu.fd),
            ("cmd_mask", gpu_out["cmd_mask"], cpu.mask),
            ("cmd_ft", gpu_out["cmd_ft"], cpu.ft),
            ("cmd_final_state", gpu_out["cmd_final_state"], cpu.final_state),
            ("cmd_j", gpu_out["cmd_j"], cpu.j),
            ("cmd_grad_ad", gpu_out["cmd_grad_ad"], grad_cpu),
        ]
        for name, g, c in checks:
            diff = float(np.max(np.abs(np.asarray(g) - np.asarray(c))))
            print(f"[E4] T18 GPU {name}: max |cpu-gpu| = {diff:.3e}")
            assert diff < TOL
        assert abs(float(gpu_out["cmd_j_mean"]) - cpu.j_mean) < TOL

    def test_cmd_three_way_on_gpu(self, gpu_out):
        """T18 GPU leg: in-worker three-way agreement, rel < 1e-5."""
        slots = gpu_out["cmd_slots"]
        ad = np.array([gpu_out["cmd_grad_ad"][b, n] for b, n in slots])
        jvp = gpu_out["cmd_grad_jvp"]
        fd = gpu_out["cmd_grad_fd"]
        rel = lambda x, y: np.abs(x - y) / np.maximum(np.abs(y), 1e-300)  # noqa: E731
        print(f"[E4] T18 GPU slots={slots.tolist()} AD={ad} JVP={jvp} FD={fd}")
        print(f"     rel AD/FD={rel(ad, fd)}, rel JVP/FD={rel(jvp, fd)}, "
              f"rel AD/JVP={rel(ad, jvp)}")
        print(f"     active-set stable={gpu_out['cmd_stable']}")
        assert np.all(gpu_out["cmd_stable"])
        assert np.all(np.isfinite(ad)) and np.all(np.isfinite(jvp))
        # fully saturated slot (last): all three legs exact zero
        assert ad[-1] == 0.0 and jvp[-1] == 0.0 and fd[-1] == 0.0
        assert np.all(rel(ad, fd) < 1.0e-5)
        assert np.all(rel(jvp, fd) < 1.0e-5)
        assert np.all(rel(ad, jvp) < 1.0e-5)
