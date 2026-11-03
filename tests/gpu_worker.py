"""D4 + E4 GPU worker: batched model II on CUDA (run as a script, not pytest).

Quadrants binds the backend once at qd.init; this script calls
ensure_init(arch=qd.cuda) BEFORE importing any skyhook model module so all
model fields are placed on the CUDA runtime. It then runs the shared D4
batch fixture (tests/d4_fixture.py):

  - forward batch rollout  -> state / a2 / fd / mask / j / j_mean
  - reverse Tape on j_mean -> full dJ_mean/dgamma_sky gradient vector
  - forward JVP (seed e_i) and central FD (h = 1e-4) for i in {0, 3, 6}

Plan B (E4) adds the same battery for the command-driven channel
(run_semi_cmd_batch on the shared FCMDS block):

  - forward batch command rollout -> cmd_* channels (mask = command
    out-of-band rate on that rollout)
  - reverse Tape on j_mean -> full (B, n_steps) adjoint dJ_mean/dfc_b
  - reverse / JVP / FD three-way at three runtime-picked slots:
    in-band, mixed-clip (record mask 1 but nonzero adjoint -- the record
    window is the stage-1 state, stages 2-4 can be in-band), and fully
    saturated (adjoint exactly zero, trajectory invariant under +/-h).

Everything is saved to an .npz (argv[1]) for the CPU-side parity tests
(tests/test_d4_gpu.py, tests/test_e4_force_batch.py; tolerance 1e-12
element-wise); the in-worker three-way summaries are printed to stdout.

Plan C (F4 / T26) adds a third battery on the model3 kernel + the
torch policy bridge:

  - policy-bridge chain: frozen-seed PolicyMLP on CUDA, mode-2 obs with
    a live preview window, bridge.sim_loss (decay 0.9) + backward ->
    policy output f, J and the bridge adjoint saved for CPU parity
    (tests/test_f4_gpu.py; tolerance 1e-9 element-wise per T26);
  - GPU in-worker bridge-adjoint three-way on the whole-window
    hinge-active fixture: reverse Tape vs FwdMode JVP vs central FD at
    three runtime-picked slots (in-band / hinge-active / saturated),
    rel < 1e-5; the full (B, N) Tape adjoint is saved for CPU parity.

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
import torch  # noqa: E402

from skyhook._runtime import ensure_init  # noqa: E402

ensure_init(arch=qd.cuda)  # bind CUDA BEFORE importing model modules

from skyhook import bridge, model2b, model3, sim3  # noqa: E402
from skyhook.sim2b import (  # noqa: E402
    _setup, _setup_fcmd, run_cmd_j_chain, run_j_chain, run_semi_batch,
    run_semi_cmd_batch)

from d4_fixture import (B, DT, FCMDS, N_STEPS, PARAMS, ROADS, X0S,  # noqa: E402
                        _Seed2D, fwd_seed_2d, perturbed, perturbed_cmd)

JVP_INDICES = (0, 3, 6)
H = 1.0e-4
T_END = N_STEPS * DT


def _run_d4_battery():
    """D4 battery: skyhook-law batch (forward + reverse + JVP + FD)."""
    res0 = run_semi_batch(PARAMS, ROADS, x0s=X0S, t_end=T_END, dt=DT)

    _setup(PARAMS, ROADS, X0S)
    with qd.ad.Tape(loss=model2b.j_mean):
        run_j_chain(N_STEPS, DT, B)
    grad_ad = model2b.pg_b.grad.to_numpy()[:B]

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
                            t_end=T_END, dt=DT)
        dn = run_semi_batch(perturbed(PARAMS, i, -H), ROADS, x0s=X0S,
                            t_end=T_END, dt=DT)
        grad_fd[i] = (up.j_mean - dn.j_mean) / (2.0 * H)
        stable[i] = (np.array_equal(res0.mask, up.mask)
                     and np.array_equal(res0.mask, dn.mask))
    return res0, grad_ad, grad_jvp, grad_fd, stable


def _run_cmd_battery():
    """E4 battery: command-driven batch (forward + reverse + JVP + FD)."""
    cmd0 = run_semi_cmd_batch(PARAMS, FCMDS, ROADS, x0s=X0S,
                              t_end=T_END, dt=DT)

    _setup(PARAMS, ROADS, X0S)
    _setup_fcmd(FCMDS, N_STEPS, B)
    with qd.ad.Tape(loss=model2b.j_mean):
        run_cmd_j_chain(N_STEPS, DT, B)
    cmd_grad = model2b.fc_b.grad.to_numpy()[:B, :N_STEPS]

    def _pick(pred):
        for b in range(B):
            for n in range(N_STEPS):
                if pred(b, n):
                    return (b, n)
        raise RuntimeError("no command slot matches the predicate")

    # in-band / mixed-clip / fully-saturated (see module docstring)
    cmd_slots = [
        _pick(lambda b, n: cmd0.mask[b, n] == 0.0 and cmd_grad[b, n] != 0.0),
        _pick(lambda b, n: cmd0.mask[b, n] == 1.0 and cmd_grad[b, n] != 0.0),
        _pick(lambda b, n: cmd0.mask[b, n] == 1.0 and cmd_grad[b, n] == 0.0),
    ]

    cmd_jvp = []
    cmd_fd = []
    cmd_stable = []
    for (b, n) in cmd_slots:
        _setup(PARAMS, ROADS, X0S)
        _setup_fcmd(FCMDS, N_STEPS, B)
        with qd.ad.FwdMode(loss=model2b.j_mean, param=model2b.fc_b,
                           seed=fwd_seed_2d(b, n)):
            run_cmd_j_chain(N_STEPS, DT, B)
        cmd_jvp.append(model2b.j_mean.dual[None])

        up = run_semi_cmd_batch(PARAMS, perturbed_cmd(FCMDS, b, n, +H),
                                ROADS, x0s=X0S, t_end=T_END, dt=DT)
        dn = run_semi_cmd_batch(PARAMS, perturbed_cmd(FCMDS, b, n, -H),
                                ROADS, x0s=X0S, t_end=T_END, dt=DT)
        cmd_fd.append((up.j_mean - dn.j_mean) / (2.0 * H))
        cmd_stable.append(bool(np.array_equal(cmd0.mask, up.mask)
                               and np.array_equal(cmd0.mask, dn.mask)))
    return cmd0, cmd_grad, cmd_slots, cmd_jvp, cmd_fd, cmd_stable


def _run_planc_battery():
    """Plan C (T26) battery: policy-bridge chain + model3 three-way on CUDA.

    Frozen-seed fixture (replicated bit-for-bit on the CPU side by
    tests/test_f4_gpu.py): B=3 mixed road families, nonzero ICs, a
    mode-2 observation whose preview window overlaps the pothole, and a
    torch.manual_seed(2026) PolicyMLP. The bridge runs the segment with
    the decay-0.9 adjoint path exercised.
    """
    from skyhook.params import SemiActiveParams
    from skyhook.policy import Obs, ObsBuffer, PolicyMLP, build_obs
    from skyhook.road import SineRoad
    from skyhook.road_seq import Event, ar_switching, events_ar, sample_analytic

    B, N, H, dt = 3, 1000, 100, 1.0e-3
    weights = sim3.DEFAULT_WEIGHTS
    _p0 = SemiActiveParams()
    params = [SemiActiveParams(gamma_sky=_p0.gamma_sky * (0.8 + 0.2 * i),
                               c_min=_p0.c_min, c_max=_p0.c_max)
              for i in range(B)]
    x0s = [(0.0, 0.01, 0.0, -0.02),
           (0.005, -0.01, 0.02, 0.03),
           (-0.005, 0.02, -0.01, 0.0)]
    xmaxs0 = [0.08] * B
    seqs = [sample_analytic(SineRoad(amp=0.02, freq_hz=2.0, ramp_t=0.5), N, dt),
            ar_switching(0.003, 0.012, 0.5, N, dt, seed=7),
            events_ar([Event(amp=-0.02, t0=0.30, width=0.08, kind="pothole"),
                       Event(amp=0.015, t0=0.60, width=0.10, kind="bump")],
                      0.003, 0.006, 0.4, N, dt, seed=11)]

    # --- policy-bridge chain (frozen seed; segment start i0 = 300 ------
    # half-steps so the 0.2 s preview window [0.15, 0.35] s sees the pothole)
    torch.manual_seed(2026)
    pol = PolicyMLP(H).cuda()
    rng = np.random.default_rng(5)
    buf = ObsBuffer(B)
    buf.push(rng.normal(0.0, 1.0, (B, 100)),
             rng.normal(0.0, 0.05, (B, 100)), dt)
    obs = build_obs(buf, seqs, 300, np.asarray(x0s), xmaxs0, dt, mode=2)
    obs = Obs(past=obs.past.cuda(), preview=obs.preview.cuda(),
              oc=obs.oc.cuda(), v_rel=obs.v_rel.cuda())
    f = pol(obs)
    f.retain_grad()      # non-leaf policy output: keep the bridge adjoint
    j = bridge.sim_loss(f, seqs, params, x0s, xmaxs=xmaxs0, dt=dt,
                        i0=300, decay=0.9)
    j.backward()
    pc = {
        "pc_policy_f": f.detach().cpu().numpy().copy(),
        "pc_policy_j": float(j.item()),
        "pc_policy_adj": f.grad.detach().cpu().numpy().copy(),
    }
    print(f"[GPU/pc] policy J={pc['pc_policy_j']:.12e} "
          f"adjoint nonzero={int(np.count_nonzero(pc['pc_policy_adj']))}"
          f"/{pc['pc_policy_adj'].size}")

    # --- whole-window bridge adjoint, hinge-active fixture --------------
    fc_rng = np.random.default_rng(123)
    fc = fc_rng.uniform(-1.0, 1.0, size=(B, N))
    fc[2, ::10] *= 8.0                 # adversarial spikes -> saturated slots
    big = sim3.run_cmd_seq_batch(params, fc, seqs, x0s=x0s,
                                 xmaxs=[10.0] * B, t_end=N * dt, dt=dt)
    dmax = np.max(np.abs(big.travel), axis=1)
    xmaxs = [10.0, 0.9 * dmax[1], 0.3 * dmax[2]]
    out = sim3.run_cmd_seq_batch(params, fc, seqs, x0s=x0s, xmaxs=xmaxs,
                                 t_end=N * dt, dt=dt)

    sim3._setup(params, x0s, xmaxs)
    sim3._setup_road(seqs, 0, N, B)
    sim3._setup_fcmd(fc, N, B)
    with qd.ad.Tape(loss=model3.j_mean3):
        sim3.run_jsim_chain(N, dt, B, 0, weights)
    g = model3.fc3.grad.to_numpy()[:B, :N].copy()
    pc["pc_full_adj"] = g
    pc["pc_j_terms"] = out.j_terms

    def pick(pred):
        for b in range(B):
            for n in range(N):
                if pred(b, n):
                    return (b, n)
        return None

    slots = {
        "inband": pick(lambda b, n: out.mask[b, n] == 0.0 and g[b, n] != 0.0),
        "hinge_active": pick(lambda b, n: b == 1 and out.mask[b, n] == 0.0
                             and g[b, n] != 0.0),
        "saturated": pick(lambda b, n: out.mask[b, n] == 1.0
                          and g[b, n] == 0.0),
    }
    assert all(s is not None for s in slots.values()), slots

    def jmean_of(fc_p):
        sim3._setup(params, x0s, xmaxs)
        sim3._setup_fcmd(fc_p, N, B)
        sim3.run_jsim_chain(N, dt, B, 0, weights)
        return model3.j_mean3[None]

    h = 1.0e-4
    rel = lambda x, y: abs(x - y) / max(abs(y), 1e-300)  # noqa: E731
    slot_names, slot_ad, slot_jvp, slot_fd = [], [], [], []
    for name in ("inband", "hinge_active", "saturated"):
        b, n = slots[name]
        seed = np.zeros((model3.B_MAX, model3.T_B_MAX), dtype=np.float64)
        seed[b, n] = 1.0
        sim3._setup(params, x0s, xmaxs)
        sim3._setup_fcmd(fc, N, B)
        with qd.ad.FwdMode(loss=model3.j_mean3, param=model3.fc3,
                           seed=seed.view(_Seed2D)):
            sim3.run_jsim_chain(N, dt, B, 0, weights)
        jvp = model3.j_mean3.dual[None]

        up = np.array(fc, copy=True); up[b, n] += h
        dn = np.array(fc, copy=True); dn[b, n] -= h
        fd = (jmean_of(up) - jmean_of(dn)) / (2.0 * h)

        print(f"[GPU/pc] slot {name} (b={b}, n={n}): AD={g[b, n]:.10e} "
              f"JVP={jvp:.10e} FD={fd:.10e} rel_AD_FD={rel(g[b, n], fd):.3e} "
              f"rel_AD_JVP={rel(g[b, n], jvp):.3e}")
        slot_names.append(name)
        slot_ad.append(g[b, n])
        slot_jvp.append(jvp)
        slot_fd.append(fd)
    pc["pc_slot_names"] = np.array(slot_names)
    pc["pc_slots"] = np.array([slots[s] for s in slot_names], dtype=np.int64)
    pc["pc_slot_ad"] = np.array(slot_ad)
    pc["pc_slot_jvp"] = np.array(slot_jvp)
    pc["pc_slot_fd"] = np.array(slot_fd)
    return pc


def main(out_path: str):
    res0, grad_ad, grad_jvp, grad_fd, stable = _run_d4_battery()
    cmd0, cmd_grad, cmd_slots, cmd_jvp, cmd_fd, cmd_stable = _run_cmd_battery()
    pc = _run_planc_battery()

    np.savez(out_path,
             state=res0.state, a2=res0.a2, fd=res0.fd, mask=res0.mask,
             ft=res0.ft, j=res0.j, j_mean=res0.j_mean,
             final_state=res0.final_state, grad_ad=grad_ad,
             jvp_indices=np.array(JVP_INDICES),
             grad_jvp=np.array([grad_jvp[i] for i in JVP_INDICES]),
             grad_fd=np.array([grad_fd[i] for i in JVP_INDICES]),
             stable=np.array([stable[i] for i in JVP_INDICES]),
             cmd_state=cmd0.state, cmd_a2=cmd0.a2, cmd_force=cmd0.fd,
             cmd_mask=cmd0.mask, cmd_ft=cmd0.ft, cmd_j=cmd0.j,
             cmd_j_mean=cmd0.j_mean, cmd_final_state=cmd0.final_state,
             cmd_grad_ad=cmd_grad,
             cmd_slots=np.array(cmd_slots, dtype=np.int64),
             cmd_grad_jvp=np.array(cmd_jvp),
             cmd_grad_fd=np.array(cmd_fd),
             cmd_stable=np.array(cmd_stable),
             **pc)

    print(f"[GPU] j_mean={res0.j_mean:.12e}")
    print(f"[GPU] grad_ad={np.round(grad_ad, 8)}")
    for i in JVP_INDICES:
        rel = abs(grad_ad[i] - grad_fd[i]) / max(abs(grad_fd[i]), 1e-300)
        print(f"[GPU] i={i}: AD={grad_ad[i]:.10e} JVP={grad_jvp[i]:.10e} "
              f"FD={grad_fd[i]:.10e} rel_ad_fd={rel:.3e} stable={stable[i]}")

    print(f"[GPU] cmd j_mean={cmd0.j_mean:.12e}")
    print(f"[GPU] cmd out-of-band rate={float(np.mean(cmd0.mask)):.4f}")
    print(f"[GPU] cmd adjoint: max|g|={float(np.max(np.abs(cmd_grad))):.6e} "
          f"nonzero={int(np.count_nonzero(cmd_grad))}/{cmd_grad.size}")
    names = ("inband", "mixed_clip", "saturated")
    for name, (b, n), jv, fdv, st in zip(names, cmd_slots, cmd_jvp,
                                         cmd_fd, cmd_stable):
        ad = cmd_grad[b, n]
        rel = lambda x, y: abs(x - y) / max(abs(y), 1e-300)  # noqa: E731
        print(f"[GPU] cmd slot {name} (b={b}, n={n}): AD={ad:.10e} "
              f"JVP={jv:.10e} FD={fdv:.10e} rel_AD_FD={rel(ad, fdv):.3e} "
              f"rel_AD_JVP={rel(ad, jv):.3e} stable={st}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/d4_gpu_out.npz")
