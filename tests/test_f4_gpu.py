"""F4 gate tests (plan C section 4, T26): GPU path closure.

tests/test_f4_gpu.py spawns the shared worker (tests/gpu_worker.py) on
CUDA and checks the plan-C chain end to end:

  T26 CPU-GPU parity, policy-bridge full chain: frozen-seed PolicyMLP
      (mode-2 obs, live preview window, bridge.sim_loss with the
      decay-0.9 adjoint path) rebuilt on CPU -- policy output f, J and
      the bridge adjoint match element-wise < 1e-9.
  T26 CPU-GPU parity, whole-window bridge adjoint: the (B, N) Tape
      adjoint on the hinge-active fixture (T19 pattern) < 1e-9.
  T26 GPU in-worker three-way: reverse Tape vs FwdMode JVP vs central
      FD at the in-band / hinge-active / saturated slots, rel < 1e-5;
      the saturated slot is exactly zero on all three routes.

Plan discipline: run pytest OUTSIDE the sandbox (CUDA needs real device
access; the worker asserts arch=cuda and fails loudly otherwise).
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import quadrants as qd
import torch

from skyhook import bridge, model3, sim3
from skyhook.params import SemiActiveParams
from skyhook.policy import ObsBuffer, PolicyMLP, build_obs
from skyhook.road import SineRoad
from skyhook.road_seq import Event, ar_switching, events_ar, sample_analytic

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "gpu_worker.py")
TOL = 1.0e-9          # T26 element-wise parity bar
REL3 = 1.0e-5         # three-way rel bar

B, N, H, DT = 3, 1000, 100, 1.0e-3
T_END = N * DT

_P0 = SemiActiveParams()
PARAMS = [SemiActiveParams(gamma_sky=_P0.gamma_sky * (0.8 + 0.2 * i),
                           c_min=_P0.c_min, c_max=_P0.c_max)
          for i in range(B)]
X0S = [(0.0, 0.01, 0.0, -0.02),
       (0.005, -0.01, 0.02, 0.03),
       (-0.005, 0.02, -0.01, 0.0)]
SEQS = [sample_analytic(SineRoad(amp=0.02, freq_hz=2.0, ramp_t=0.5), N, DT),
        ar_switching(0.003, 0.012, 0.5, N, DT, seed=7),
        events_ar([Event(amp=-0.02, t0=0.30, width=0.08, kind="pothole"),
                   Event(amp=0.015, t0=0.60, width=0.10, kind="bump")],
                  0.003, 0.006, 0.4, N, DT, seed=11)]

_FC_RNG = np.random.default_rng(123)
FC = _FC_RNG.uniform(-1.0, 1.0, size=(B, N))
FC[2, ::10] *= 8.0    # adversarial spikes -> saturated command slots


def _max_rel(a, b, floor=1.0e-6):
    """max |a-b| / max(|b|, floor): scale-aware element-wise parity."""
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


class TestT26GpuParity:
    @pytest.fixture(scope="class")
    def gpu_out(self, tmp_path_factory):
        out = str(tmp_path_factory.mktemp("f4_gpu") / "out.npz")
        r = subprocess.run([sys.executable, WORKER, out],
                           capture_output=True, text=True, timeout=3600)
        print(r.stdout)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "arch=cuda" in r.stdout, "worker did not start on CUDA"
        return np.load(out)

    def test_policy_bridge_chain_parity(self, gpu_out):
        """f / J / adjoint of the frozen-seed policy-bridge chain."""
        torch.manual_seed(2026)
        pol = PolicyMLP(H)
        rng = np.random.default_rng(5)
        buf = ObsBuffer(B)
        buf.push(rng.normal(0.0, 1.0, (B, 100)),
                 rng.normal(0.0, 0.05, (B, 100)), DT)
        obs = build_obs(buf, SEQS, 300, np.asarray(X0S), [0.08] * B, DT,
                        mode=2)
        f = pol(obs)
        f.retain_grad()    # non-leaf policy output: keep the bridge adjoint
        j = bridge.sim_loss(f, SEQS, PARAMS, X0S, xmaxs=[0.08] * B, dt=DT,
                            i0=300, decay=0.9)
        j.backward()

        rel_f = _max_rel(gpu_out["pc_policy_f"], f.detach().numpy())
        rel_j = abs(float(gpu_out["pc_policy_j"]) - float(j.item())) \
            / max(abs(float(j.item())), 1e-300)
        rel_adj = _max_rel(gpu_out["pc_policy_adj"], f.grad.detach().numpy())
        print(f"[F4] T26 chain parity: rel_f={rel_f:.3e} rel_j={rel_j:.3e} "
              f"rel_adj={rel_adj:.3e}")
        assert rel_f < TOL
        assert rel_j < TOL
        assert rel_adj < TOL
        # the chain is alive on both devices: nonzero adjoint mass
        assert np.count_nonzero(gpu_out["pc_policy_adj"]) > 0

    def test_full_window_adjoint_parity(self, gpu_out):
        """Whole-window Tape adjoint on the hinge-active fixture."""
        big = sim3.run_cmd_seq_batch(PARAMS, FC, SEQS, x0s=X0S,
                                     xmaxs=[10.0] * B, t_end=T_END, dt=DT)
        dmax = np.max(np.abs(big.travel), axis=1)
        xmaxs = [10.0, 0.9 * dmax[1], 0.3 * dmax[2]]
        out = sim3.run_cmd_seq_batch(PARAMS, FC, SEQS, x0s=X0S, xmaxs=xmaxs,
                                     t_end=T_END, dt=DT)
        # fixture sanity: elements 1/2 hinge-active on both devices
        assert out.j_terms[1, 3] > 0.0 and out.j_terms[2, 3] > 0.0

        sim3._setup(PARAMS, X0S, xmaxs)
        sim3._setup_road(SEQS, 0, N, B)
        sim3._setup_fcmd(FC, N, B)
        with qd.ad.Tape(loss=model3.j_mean3):
            sim3.run_jsim_chain(N, DT, B, 0, sim3.DEFAULT_WEIGHTS)
        g_cpu = model3.fc3.grad.to_numpy()[:B, :N]

        rel = _max_rel(gpu_out["pc_full_adj"], g_cpu)
        print(f"[F4] T26 full-window adjoint parity: rel={rel:.3e}")
        assert rel < TOL

    def test_gpu_three_way(self, gpu_out):
        """In-worker Tape vs JVP vs FD at the three semantic slots."""
        names = [str(s) for s in gpu_out["pc_slot_names"]]
        ad = gpu_out["pc_slot_ad"]
        jvp = gpu_out["pc_slot_jvp"]
        fd = gpu_out["pc_slot_fd"]
        rel = lambda x, y: abs(x - y) / max(abs(y), 1e-300)  # noqa: E731
        for i, name in enumerate(names):
            print(f"[F4] T26 slot {name}: AD={ad[i]:.10e} JVP={jvp[i]:.10e} "
                  f"FD={fd[i]:.10e} rel_AD_FD={rel(ad[i], fd[i]):.3e} "
                  f"rel_AD_JVP={rel(ad[i], jvp[i]):.3e}")
            if name == "saturated":
                assert ad[i] == 0.0 and jvp[i] == 0.0 and fd[i] == 0.0
            else:
                assert rel(ad[i], fd[i]) < REL3
                assert rel(ad[i], jvp[i]) < REL3
        assert set(names) == {"inband", "hinge_active", "saturated"}
        # hinge slots really were hinge-active in the GPU run too
        assert gpu_out["pc_j_terms"][1, 3] > 0.0
        assert gpu_out["pc_j_terms"][2, 3] > 0.0
