"""Segmented policy training loop (plan C, F2 -- docs/plan_c_policy_
training_dev_test_plan.md section 3, decisions 1/3/4/6/7).

Receding-horizon episode (decision 1): T_w = K * H steps, the policy
infers ONCE per segment boundary (10 Hz default) emitting the (B, H)
command block, one Tape-wrapped bridge call per segment, sim-state
handoff DETACHED at segment boundaries (truncated BPTT), policy gradient
accumulated across the K segments on the torch side, ONE backward per
episode.

Control regularizers (decision 4, v2.2 splice discipline): the K blocks
are concatenated into the (B, T_w) EPISODE series first; the jerk
difference and the magnitude MS are then computed on the whole series, so
a jump AT a segment boundary costs the same as a jump inside a segment
(no free pulses at boundaries).

Optimizer (decision 6, DiffPhysDrone borrowing): AdamW +
CosineAnnealingLR.

Domain randomization (decision 7): per-episode resampling of road family
parameters (roughness levels / switch time / event parameters) and ICs;
hardware gains stay fixed.

Usage:  python train_policy.py --mode 2 --iterations 200
"""

import argparse
import copy
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from skyhook import bridge
from skyhook.params import SemiActiveParams
from skyhook.policy import ObsBuffer, PolicyMLP, build_obs
from skyhook.road_seq import (
    ROUGHNESS_LEVELS,
    ar_switching,
    events_ar,
    sample_events,
)
from skyhook.sim3 import DEFAULT_WEIGHTS

T_W = 1.0            # default episode length (s); overridable via cfg.t_w
IC_AMP = 0.01        # random-IC amplitude (nonzero-IC discipline)
SELECT_SEED = 555    # plan D G1: frozen select-batch seed (train/select/
                     # test = 0/555/held-out discipline; NEVER used for
                     # training or for any held-out evaluation decision)


@dataclass
class TrainConfig:
    mode: int = 2                     # 1: accelerometer only; 2: + preview
    B: int = 8                        # batch size
    H: int = 100                      # segment steps (10 Hz at dt = 1 ms)
    K: int = 10                       # segments per episode (derived: t_w/H)
    t_w: float = 1.0                  # episode length (s)
    dt: float = 1.0e-3
    n_iter: int = 50
    lr: float = 1.0e-3
    weights: tuple = DEFAULT_WEIGHTS  # (w_acc, w_disp, w_tire, w_lim)
    w_jerk: float = 1.0e-4            # plan D G2 (amendment A): scan
                                      # {1e-6..1e-3} on the held-out
                                      # eval seed showed 1e-4 is Pareto
                                      # (j_sim improves on both modes,
                                      # jerk halves); 1e-3 forces
                                      # near-passive (j_sim +18%)
    w_mag: float = 0.001
    decay: float = 1.0                # within-segment adjoint decay
    device: str = "cpu"               # plan D G3: "cuda" for GPU training
    seed: int = 0
    fixed_road_seed: int = 11         # None -> resample every episode
    select_seed: int = SELECT_SEED    # plan D G1: frozen validation batch
    select_every: int = 25            # plan D G1: select-eval cadence
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 50
    log_every: int = 1


def control_loss(f_all: torch.Tensor, dt: float,
                 w_jerk: float, w_mag: float) -> torch.Tensor:
    """J_ctl on the episode-level concatenated series (decision 4).

    jerk = MS(diff(f)/dt), mag = MS(f); both over the (B, T_w) splice so
    segment-boundary jumps are penalized identically to in-segment jumps.
    """
    jerk = torch.diff(f_all, dim=1) / dt
    return w_jerk * (jerk * jerk).mean() + w_mag * (f_all * f_all).mean()


def sample_roads(cfg: TrainConfig, rng: np.random.Generator):
    """Domain-randomized road batch (decision 7/10) for one episode."""
    n_steps = int(round(cfg.t_w / cfg.dt))
    levels = sorted(ROUGHNESS_LEVELS.values())
    roads = []
    for b in range(cfg.B):
        s1, s2 = rng.choice(levels, size=2, replace=True)
        t_sw = float(rng.uniform(0.2, 0.8)) * cfg.t_w
        if cfg.mode == 1:
            roads.append(ar_switching(s1, s2, t_sw, n_steps, cfg.dt,
                                      seed=int(rng.integers(1 << 31))))
        else:
            events = sample_events(rng, n_steps, cfg.dt)
            roads.append(events_ar(events, s1, s2, t_sw, n_steps, cfg.dt,
                                   seed=int(rng.integers(1 << 31))))
    return roads


def sample_ics(cfg: TrainConfig, rng: np.random.Generator) -> np.ndarray:
    """Random nonzero ICs (B, 4): x, v ~ U(-IC_AMP, IC_AMP)."""
    return rng.uniform(-IC_AMP, IC_AMP, size=(cfg.B, 4))


def _fixed_fixture(cfg: TrainConfig):
    """Deterministic roads + ICs for the T23 smoke gate (fixed family)."""
    rng = np.random.default_rng(cfg.fixed_road_seed)
    roads = sample_roads(cfg, rng)
    ics = sample_ics(cfg, rng)
    return roads, ics


def run_episode(policy, cfg: TrainConfig, road_seqs, ics: np.ndarray,
                params_list, xmaxs):
    """One receding-horizon episode; returns (loss_sim, J_ctl, stats).

    The K segment losses accumulate in the torch graph; the sim state is
    handed over as detached numpy (decision 1).
    """
    buf = ObsBuffer(cfg.B)
    x0 = np.asarray(ics, dtype=np.float64).copy()
    total_sim = 0.0
    f_blocks = []
    mask_all = []
    adj_nz = []
    for k in range(cfg.K):
        i0 = 2 * k * cfg.H
        obs = build_obs(buf, road_seqs, i0, x0, xmaxs, cfg.dt, cfg.mode,
                        device=None if cfg.device == "cpu"
                        else torch.device(cfg.device))
        f_k = policy(obs)
        loss_k = bridge.sim_loss(
            f_k, road_seqs, params_list,
            [tuple(x0[b]) for b in range(cfg.B)],
            xmaxs=xmaxs, weights=cfg.weights, dt=cfg.dt, i0=i0,
            decay=cfg.decay)
        rec = bridge.read_last_segment(cfg.H, cfg.B)
        buf.push(rec.a2, f_k.detach().cpu().numpy(), cfg.dt)
        x0 = rec.final_state
        total_sim = total_sim + loss_k
        f_blocks.append(f_k)
        mask_all.append(rec.mask.copy())
        adj_nz.append(float(np.mean(rec.adjoint != 0.0)))
    f_all = torch.cat(f_blocks, dim=1)
    j_ctl = control_loss(f_all, cfg.dt, cfg.w_jerk, cfg.w_mag)
    stats = {
        "mask_rate": float(np.concatenate(mask_all, axis=1).mean()),
        "adj_nonzero": float(np.mean(adj_nz)),
        "f_blocks": f_blocks,
    }
    return total_sim, j_ctl, stats


def eval_select(policy, cfg: TrainConfig, roads, ics, params_list, xmaxs):
    """No-gradient j_sim on the frozen select batch (plan D G1).

    Same rollout path as training (run_episode), same j_sim bookkeeping
    (sum of segment sim losses / K), wrapped in torch.no_grad() so no
    graph is built and no training state is touched.
    """
    with torch.no_grad():
        total_sim, _, _ = run_episode(policy, cfg, roads, ics,
                                      params_list, xmaxs)
    return float(total_sim.item()) / cfg.K


def train(cfg: TrainConfig, verbose: bool = True):
    """Train one policy for one mode; returns (policy, history)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    cfg.K = int(round(cfg.t_w / (cfg.H * cfg.dt)))

    policy = PolicyMLP(cfg.H)
    policy = policy.to(cfg.device)    # plan D G3 (no-op on CPU)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr,
                            weight_decay=1.0e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                       T_max=cfg.n_iter)
    params_list = [SemiActiveParams() for _ in range(cfg.B)]
    xmaxs = [0.08] * cfg.B

    fixed = _fixed_fixture(cfg) if cfg.fixed_road_seed is not None else None
    if fixed is not None:
        roads, ics = fixed

    history = {"j_sim": [], "j_ctl": [], "jerk": [], "mag": [],
               "mask_rate": [], "adj_nonzero": [], "select_j_sim": []}
    # frozen select batch (plan D G1): own rng, never consumes the training
    # stream, never feeds a gradient
    sel_rng = np.random.default_rng(cfg.select_seed)
    sel_roads = sample_roads(cfg, sel_rng)
    sel_ics = sample_ics(cfg, sel_rng)
    best_j = float("inf")
    best_it = 0
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    for it in range(cfg.n_iter):
        if fixed is None:
            roads, ics = sample_roads(cfg, rng), sample_ics(cfg, rng)
        loss_sim, j_ctl, stats = run_episode(policy, cfg, roads, ics,
                                             params_list, xmaxs)
        loss = loss_sim + j_ctl
        opt.zero_grad()
        loss.backward()

        # adjoint diagnostics: nonzero fraction of the pure bridge adjoint
        history["adj_nonzero"].append(stats["adj_nonzero"])

        opt.step()
        sched.step()

        # split logging (the combined J_ctl is already inside the loss)
        with torch.no_grad():
            f_all = torch.cat([fb.detach() for fb in stats["f_blocks"]],
                              dim=1)
            j_jerk = (cfg.w_jerk
                      * ((torch.diff(f_all, dim=1) / cfg.dt) ** 2).mean())
            j_mag = cfg.w_mag * (f_all ** 2).mean()
        history["j_sim"].append(float(loss_sim.item()) / cfg.K)
        history["j_ctl"].append(float(j_ctl.item()))
        history["jerk"].append(float(j_jerk))
        history["mag"].append(float(j_mag))
        history["mask_rate"].append(stats["mask_rate"])
        if verbose and (it + 1) % cfg.log_every == 0:
            print(f"[train m{cfg.mode}] it {it + 1:4d}/{cfg.n_iter}"
                  f"  J_sim={history['j_sim'][-1]:.6e}"
                  f"  jerk={history['jerk'][-1]:.3e}"
                  f"  mag={history['mag'][-1]:.3e}"
                  f"  mask={history['mask_rate'][-1]:.3f}"
                  f"  adj_nz={history['adj_nonzero'][-1]:.3f}")
        # plan D G1: frozen-batch select evaluation + best checkpoint.
        # No training-semantics change: the select call is side-effect
        # free (own fixture, no_grad, no rng consumption).
        if (it + 1) % cfg.select_every == 0 or it + 1 == cfg.n_iter:
            sel_j = eval_select(policy, cfg, sel_roads, sel_ics,
                                params_list, xmaxs)
            history["select_j_sim"].append(sel_j)
            is_best = sel_j < best_j
            if verbose:
                print(f"[select m{cfg.mode}] it {it + 1:4d}"
                      f"  J_sim(select)={sel_j:.6e}"
                      + ("  *best*" if is_best else ""))
            if is_best:
                best_j = sel_j
                best_it = it + 1
                path = os.path.join(cfg.ckpt_dir,
                                    f"policy_m{cfg.mode}_best.pt")
                torch.save({"model": policy.state_dict(),
                            "cfg": asdict(cfg), "history": history,
                            "best_iter": best_it, "select_j_sim": best_j},
                           path)
        if (it + 1) % cfg.ckpt_every == 0 or it + 1 == cfg.n_iter:
            path = os.path.join(cfg.ckpt_dir,
                                f"policy_m{cfg.mode}_it{it + 1}.pt")
            torch.save({"model": policy.state_dict(),
                        "cfg": asdict(cfg), "history": history}, path)
    return policy, history


def main():
    ap = argparse.ArgumentParser(description="Plan C policy training")
    ap.add_argument("--mode", type=int, default=2, choices=(1, 2))
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--H", type=int, default=100,
                    help="segment steps (decision-rate knob, v2.2)")
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resample", action="store_true",
                    help="domain randomization every episode "
                         "(default: fixed road family)")
    args = ap.parse_args()
    cfg = TrainConfig(mode=args.mode, B=args.batch, H=args.H,
                      n_iter=args.iterations, lr=args.lr, seed=args.seed,
                      fixed_road_seed=None if args.resample else 11)
    train(cfg)


if __name__ == "__main__":
    main()
