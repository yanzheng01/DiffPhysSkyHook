"""Per-mode policy evaluation + baseline comparison (plan C, F3 -- docs/
plan_c_policy_training_dev_test_plan.md section 3, F3; gates T25/T28).

Protocols:
  mode 1 (roughness switching): six-metric table PLUS the switch-settling
    time and the pre/post-switch split J_acc (the adaptation metric).
  mode 2 (discrete events): six-metric table PLUS the event-window peak
    |v_dot_2| (event time +-0.2 s) and the event displacement overshoot.

Controllers compared on the SAME (held-out-seed) road batch / ICs:
  - passive c_min        (1 kHz state feedback, Plan B semantics)
  - continuous skyhook   (1 kHz, gamma_sky * v2 through the device clip)
  - ON-OFF skyhook       (1 kHz, hard switch on v2 * v_rel)
  - trained policy       (sub = H -> 10 Hz segment-boundary decisions,
                          exactly the training-time information structure)
  - open-loop PGD upper bound    (whole-window f optimization on pure
    J_sim; PRIVILEGED: full preview, non-causal -- the v1.x one-shot
    controller demoted to an upper bound, plan C decision 1)
  - privileged planner   (same whole-window optimization with the
    training J_ctl regularizers; PRIVILEGED likewise)

Closed-loop roller: one road write + one _setup per episode, then per
segment cmd_fn -> _setup_fcmd -> run_jsim_chain (NO Tape -- evaluation
needs no gradients) -> record snapshot -> set_state_hist3 handoff. The
roller bypasses bridge.sim_loss entirely but shares its road-write cache
discipline: the bridge cache is invalidated on entry/exit so a direct
sim3._setup_road elsewhere can never leave the cache stale.

Usage:  python eval_policy.py --mode 2 [--ckpt path] [--iterations 200]
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch

from skyhook import bridge, model3, sim3
from skyhook.params import SemiActiveParams
from skyhook.policy import ObsBuffer, PolicyMLP, build_obs
from train_policy import TrainConfig, control_loss, sample_ics, sample_roads, train


# ---------------------------------------------------------------------------
# closed-loop roller
# ---------------------------------------------------------------------------
@dataclass
class EpisodeRecords:
    """Whole-episode whole-step records of one closed-loop rollout."""

    t: np.ndarray            # (N,) global time grid t_n = n*dt
    state: np.ndarray        # (B, N, 4)
    a2: np.ndarray           # (B, N)
    fd: np.ndarray           # (B, N) realized device force
    mask: np.ndarray         # (B, N) command out-of-band flag
    ft: np.ndarray           # (B, N) tire force
    travel: np.ndarray       # (B, N) suspension stroke x2 - x1
    f_cmd: np.ndarray        # (B, N) REQUESTED command series
    j_sim: np.ndarray        # (B,) episode composite loss (segment mean)
    j_terms: np.ndarray      # (B, 4) MS terms (acc, disp, tire, hinge)


def _segment_snapshot(sub: int, B: int, f_seg: np.ndarray):
    """Read model3 fields after one run_jsim_chain (like SeqRunResult)."""
    return {
        "state": model3.traj3.to_numpy()[:B, :sub].copy(),
        "a2": model3.acc3.to_numpy()[:B, :sub].copy(),
        "fd": model3.fd3.to_numpy()[:B, :sub].copy(),
        "mask": model3.mask3.to_numpy()[:B, :sub].copy(),
        "ft": model3.ft3.to_numpy()[:B, :sub].copy(),
        "travel": model3.travel3.to_numpy()[:B, :sub].copy(),
        "final_state": model3.xh3.to_numpy()[:B, sub].copy(),
        "j_sim": model3.j_val3.to_numpy()[:B].copy(),
        "j_terms": np.stack([
            model3.js_acc3.to_numpy()[:B] / sub,
            model3.js_disp3.to_numpy()[:B] / sub,
            model3.js_tire3.to_numpy()[:B] / sub,
            model3.js_lim3.to_numpy()[:B] / sub,
        ], axis=1),
        "f_cmd": f_seg,
    }


def closed_loop_rollout(cmd_fn, road_seqs, params_list, x0s, xmaxs=None,
                        n_steps: int = 1000, dt: float = 1.0e-3,
                        sub: int = 1, weights=sim3.DEFAULT_WEIGHTS):
    """Closed-loop batched rollout with a `sub`-step command period.

    cmd_fn(k, state, last) -> (B, sub) command block for segment k
    (k = 0..n_steps//sub-1), where `state` is the (B, 4) boundary state and
    `last` the previous segment's snapshot dict (None before the first
    segment -- the cold-start convention of the policy buffer). Baselines
    use sub=1 (1 kHz state feedback, Plan B semantics); the policy uses
    sub=H (the 10 Hz training decision rate); open-loop oracles replay a
    fixed series with sub=n_steps.

    The whole-episode road grid is written ONCE (bridge cache invalidated
    on entry and exit -- see the module docstring).
    """
    B = len(params_list)
    if n_steps % sub != 0:
        raise ValueError(f"n_steps={n_steps} not divisible by sub={sub}")
    n_sub = n_steps // sub
    if xmaxs is None:
        xmaxs = [sim3.X_MAX_DEFAULT] * B
    x0s = np.asarray(x0s, dtype=np.float64)

    bridge._road_cache["key"] = None      # entry invalidation
    sim3._setup(params_list, [tuple(x0s[b]) for b in range(B)], xmaxs)
    sim3._setup_road(road_seqs, 0, n_steps, B)

    state = x0s.copy()
    last = None
    keys = ("state", "a2", "fd", "mask", "ft", "travel", "f_cmd")
    cat = {k: [] for k in keys}
    j_sim = []
    j_terms = []
    for k in range(n_sub):
        f = np.asarray(cmd_fn(k, state, last), dtype=np.float64)
        if f.shape != (B, sub):
            raise ValueError(f"cmd_fn returned {f.shape}, expected ({B}, {sub})")
        sim3._setup_fcmd(f, sub, B)
        sim3.run_jsim_chain(sub, dt, B, 2 * k * sub, weights)
        snap = _segment_snapshot(sub, B, f)
        for b in range(B):                 # Markov handoff (T27 pattern)
            model3.set_state_hist3(b, *snap["final_state"][b])
        state = snap["final_state"]
        last = snap
        for key in keys:
            cat[key].append(snap[key])
        j_sim.append(snap["j_sim"])
        j_terms.append(snap["j_terms"])
    bridge._road_cache["key"] = None      # exit invalidation

    out = {key: np.concatenate(v, axis=1) for key, v in cat.items()}
    return EpisodeRecords(
        t=np.arange(n_steps, dtype=np.float64) * dt,
        state=out["state"], a2=out["a2"], fd=out["fd"], mask=out["mask"],
        ft=out["ft"], travel=out["travel"], f_cmd=out["f_cmd"],
        j_sim=np.mean(np.stack(j_sim), axis=0),
        j_terms=np.mean(np.stack(j_terms), axis=0),
    )


# ---------------------------------------------------------------------------
# controller factories
# ---------------------------------------------------------------------------
def make_passive(params_list):
    """f = c_min * v_rel: the passive lower-bound damper (1 kHz, sub=1)."""
    cmin = np.array([p.c_min for p in params_list])

    def cmd_fn(k, state, last):
        vr = state[:, 3] - state[:, 2]
        return (cmin * vr)[:, None]
    return cmd_fn


def make_skyhook(params_list):
    """Continuous skyhook: f = gamma_sky * v2 (device clip enforces the
    force window; the command itself is the ideal skyhook force)."""
    gsky = np.array([p.gamma_sky for p in params_list])

    def cmd_fn(k, state, last):
        return (gsky * state[:, 3])[:, None]
    return cmd_fn


def make_onoff(params_list):
    """ON-OFF skyhook: full gamma_sky*v2 when dissipative (v2*v_rel > 0),
    else the passive c_min*v_rel floor."""
    gsky = np.array([p.gamma_sky for p in params_list])
    cmin = np.array([p.c_min for p in params_list])

    def cmd_fn(k, state, last):
        v2 = state[:, 3]
        vr = v2 - state[:, 2]
        return np.where(v2 * vr > 0.0, gsky * v2, cmin * vr)[:, None]
    return cmd_fn


def make_policy_cmd(policy, road_seqs, xmaxs, dt: float, mode: int,
                    sub: int):
    """cmd_fn wrapping a trained PolicyMLP at the training decision rate.

    Mirrors train_policy.run_episode exactly: at boundary k the past
    buffer already holds segments 0..k-1 (pushed from `last`), the
    observation is built at the global offset i0 = 2*k*sub, and the
    command block is emitted under no_grad (evaluation only).
    """
    B = len(road_seqs)
    buf = ObsBuffer(B)

    def cmd_fn(k, state, last):
        if last is not None:
            buf.push(last["a2"], last["f_cmd"], dt)
        obs = build_obs(buf, road_seqs, 2 * k * sub, state, xmaxs, dt, mode)
        with torch.no_grad():
            f = policy(obs)
        return f.detach().cpu().numpy()
    return cmd_fn


def make_openloop_cmd(f_full, sub: int):
    """cmd_fn replaying a fixed (B, N) command series (oracle evaluation)."""
    f_full = np.asarray(f_full, dtype=np.float64)

    def cmd_fn(k, state, last):
        return f_full[:, k * sub:(k + 1) * sub]
    return cmd_fn


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def six_metrics(rec: EpisodeRecords, dt: float) -> dict:
    """The plan-C six-metric table row (plus the composite J_sim).

    J_acc, RMS(v2), MS/max stroke, MS(F_t), command out-of-band rate,
    jerk = MS(diff(f_cmd)/dt) on the episode series.
    """
    a2 = rec.a2
    v2 = rec.state[:, :, 3]
    f = rec.f_cmd
    return {
        "J_acc": float(np.mean(a2 * a2)),
        "rms_v2": float(np.sqrt(np.mean(v2 * v2))),
        "ms_travel": float(np.mean(rec.travel * rec.travel)),
        "max_travel": float(np.max(np.abs(rec.travel))),
        "ms_ft": float(np.mean(rec.ft * rec.ft)),
        "mask_rate": float(np.mean(rec.mask)),
        "jerk": float(np.mean((np.diff(f, axis=1) / dt) ** 2)),
        "j_sim": float(np.mean(rec.j_sim)),
    }


def mode1_metrics(rec: EpisodeRecords, road_seqs, dt: float,
                  win_ms: float = 50.0, band: float = 0.10,
                  tail_s: float = 0.2) -> dict:
    """Mode-1 protocol: switch settling + pre/post-switch split J_acc.

    Settling time (per batch element, then batch mean): with w(n) the
    win_ms-windowed MS(a2) starting at step n, the post-switch steady
    level L is the mean of w over windows starting in the last tail_s of
    the episode; the settling time is the span from t_switch to the step
    after the LAST window (fully post-switch, n >= n_sw) that exceeds
    (1+band)*L -- i.e. when the response enters and stays inside the
    +-band of its own post-switch steady level. 0.0 if it never exceeds.
    """
    B, N = rec.a2.shape
    win = int(round(win_ms * 1.0e-3 / dt))
    if win < 1 or N <= win:
        raise ValueError("episode too short for the settling window")
    tail_n = int(round(tail_s / dt))
    settle = []
    pre_j = []
    post_j = []
    for b in range(B):
        t_sw = float(road_seqs[b].meta["t_switch"])
        n_sw = int(round(t_sw / dt))
        w2 = rec.a2[b] * rec.a2[b]
        cum = np.concatenate([[0.0], np.cumsum(w2)])
        wms = (cum[win:] - cum[:-win]) / win          # index n = window start
        pre_j.append(float(w2[:max(n_sw, 1)].mean()))
        post_j.append(float(w2[n_sw:].mean()) if n_sw < N else 0.0)
        lo = max(n_sw, N - win - tail_n)
        level = float(wms[lo:].mean())
        viol = np.nonzero(wms[n_sw:N - win + 1] > (1.0 + band) * level)[0]
        if viol.size == 0:
            settle.append(0.0)
        else:
            t_set = (n_sw + viol[-1] + 1) * dt - t_sw
            settle.append(float(min(max(t_set, 0.0), N * dt - t_sw)))
    return {"settling_s": float(np.mean(settle)),
            "settling_all": settle,
            "pre_J_acc": float(np.mean(pre_j)),
            "post_J_acc": float(np.mean(post_j))}


def mode2_metrics(rec: EpisodeRecords, road_seqs, dt: float,
                  pre_s: float = 0.2, post_s: float = 0.2) -> dict:
    """Mode-2 protocol: event-window peak |v_dot_2| + displacement overshoot.

    Event window = [t0 - pre_s, t0 + width + post_s] (clipped to the
    episode). Peak = max |a2| in the window; overshoot = max |x2(t) -
    x2(window start)| (the body excursion caused by the event, relative
    to the pre-event position). Reported per event, then averaged.
    """
    B, N = rec.a2.shape
    peaks = []
    overs = []
    for b in range(B):
        for (_kind, _amp, t0, width) in road_seqs[b].meta["events"]:
            n0 = max(0, int(round((t0 - pre_s) / dt)))
            n1 = min(N, int(round((t0 + width + post_s) / dt)))
            if n1 - n0 < 2:
                continue
            peaks.append(float(np.max(np.abs(rec.a2[b, n0:n1]))))
            x_ref = rec.state[b, n0, 1]
            overs.append(float(np.max(np.abs(rec.state[b, n0:n1, 1] - x_ref))))
    if not peaks:
        raise ValueError("no events found in road meta (mode 2 protocol)")
    return {"event_peak_acc": float(np.mean(peaks)),
            "event_overshoot_m": float(np.mean(overs)),
            "event_peak_all": peaks}


# ---------------------------------------------------------------------------
# whole-window oracles (privileged)
# ---------------------------------------------------------------------------
def oracle_optimize(road_seqs, params_list, x0s, xmaxs=None,
                    weights=sim3.DEFAULT_WEIGHTS, dt: float = 1.0e-3,
                    w_jerk: float = 0.0, w_mag: float = 0.0,
                    n_iter: int = 100, lr: float = 5.0e-2,
                    verbose: bool = True):
    """Whole-window open-loop f optimization through the bridge.

    w_jerk = w_mag = 0 -> the PGD upper bound (pure J_sim); nonzero ->
    the privileged planner (training's J_ctl regularizers). PRIVILEGED:
    the optimizer sees the whole road/episode at once (full preview,
    non-causal) -- plan C decision 1 demotes the v1.x one-shot controller
    to exactly this role.

    Initialization: the midpoint of the force window along the PASSIVE
    trajectory, 0.5*(c_min+c_max)*v_rel(t). The zero command sits on the
    out-of-band plateau where the adjoint is exactly zero (T19 saturated
    slots), and the passive floor rides the clip kink -- the midpoint is
    strictly inside the window, so the gradient flows from step one.
    """
    B = len(params_list)
    N = road_seqs[0].n_steps
    x0s = np.asarray(x0s, dtype=np.float64)
    if xmaxs is None:
        xmaxs = [sim3.X_MAX_DEFAULT] * B

    passive = closed_loop_rollout(make_passive(params_list), road_seqs,
                                  params_list, x0s, xmaxs, N, dt, sub=1,
                                  weights=weights)
    vr = passive.state[:, :, 3] - passive.state[:, :, 2]
    cmid = np.array([0.5 * (p.c_min + p.c_max) for p in params_list])
    f = torch.tensor(cmid[:, None] * vr, dtype=torch.float64)
    f.requires_grad_(True)
    opt = torch.optim.AdamW([f], lr=lr)
    x0_tuples = [tuple(x0s[b]) for b in range(B)]
    for it in range(n_iter):
        opt.zero_grad()
        loss = bridge.sim_loss(f, road_seqs, params_list, x0_tuples,
                               xmaxs=xmaxs, weights=weights, dt=dt, i0=0)
        if w_jerk or w_mag:
            loss = loss + control_loss(f, dt, w_jerk, w_mag)
        loss.backward()
        opt.step()
        if verbose and ((it + 1) % 25 == 0 or it == 0):
            print(f"[oracle{'+' if (w_jerk or w_mag) else ''}] "
                  f"it {it + 1:4d}/{n_iter}  J={float(loss.item()):.6e}")
    return f.detach().cpu().numpy().copy()


# ---------------------------------------------------------------------------
# evaluation driver
# ---------------------------------------------------------------------------
def build_eval_fixture(mode: int, B: int, t_w: float, dt: float, seed: int):
    """Held-out road batch + ICs (independent of the training seed)."""
    like = TrainConfig(mode=mode, B=B, t_w=t_w, dt=dt)
    rng = np.random.default_rng(seed)
    roads = sample_roads(like, rng)
    ics = sample_ics(like, rng)
    return roads, ics


METRIC_COLS = ("j_sim", "J_acc", "rms_v2", "ms_travel", "max_travel",
               "ms_ft", "mask_rate", "jerk")


def _print_table(title: str, rows, cols=METRIC_COLS):
    print(f"\n=== {title} ===")
    head = f"{'controller':<28}" + "".join(f"{c:>12}" for c in cols)
    print(head)
    for row in rows:
        line = f"{row['name']:<28}"
        for c in cols:
            line += f"{row[c]:>12.4g}"
        print(line)


def evaluate_controller(name, cmd_fn, sub, roads, params_list, ics, xmaxs,
                        mode: int, n_steps: int, dt: float,
                        weights=sim3.DEFAULT_WEIGHTS) -> dict:
    """One table row: rollout + six metrics + mode-specific extras."""
    rec = closed_loop_rollout(cmd_fn, roads, params_list, ics, xmaxs,
                              n_steps, dt, sub, weights)
    row = {"name": name, "sub": sub}
    row.update(six_metrics(rec, dt))
    if mode == 1:
        row.update(mode1_metrics(rec, roads, dt))
    else:
        row.update(mode2_metrics(rec, roads, dt))
    return row


def run_eval(cfg: TrainConfig, policy=None, eval_seed: int = 2026,
             oracle_iters: int = 100, oracle_lr: float = 5.0e-2,
             verbose: bool = True):
    """Full per-mode evaluation: baselines + policy + privileged oracles.

    Returns (rows, recs): the table rows and the policy's EpisodeRecords
    (for T28's per-event / settling comparisons).
    """
    n_steps = int(round(cfg.t_w / cfg.dt))
    roads, ics = build_eval_fixture(cfg.mode, cfg.B, cfg.t_w, cfg.dt,
                                    eval_seed)
    params_list = [SemiActiveParams() for _ in range(cfg.B)]
    xmaxs = [0.08] * cfg.B

    if policy is None:
        if verbose:
            print(f"[eval] training mode-{cfg.mode} policy "
                  f"({cfg.n_iter} iterations, seed {cfg.seed})")
        policy, _hist = train(cfg, verbose=verbose)

    eval_args = dict(roads=roads, params_list=params_list, ics=ics,
                     xmaxs=xmaxs, mode=cfg.mode, n_steps=n_steps,
                     dt=cfg.dt, weights=cfg.weights)
    rows = [
        evaluate_controller("passive c_min",
                            make_passive(params_list), 1, **eval_args),
        evaluate_controller("continuous skyhook",
                            make_skyhook(params_list), 1, **eval_args),
        evaluate_controller("ON-OFF skyhook",
                            make_onoff(params_list), 1, **eval_args),
    ]
    policy_fn = make_policy_cmd(policy, roads, xmaxs, cfg.dt, cfg.mode,
                                cfg.H)
    rows.append(evaluate_controller(f"policy (mode {cfg.mode}, "
                                    f"{cfg.H} steps)", policy_fn, cfg.H,
                                    **eval_args))
    recs = closed_loop_rollout(policy_fn, roads, params_list, ics, xmaxs,
                               n_steps, cfg.dt, cfg.H, cfg.weights)

    # privileged whole-window oracles (upper bounds, marked as such)
    f_pgd = oracle_optimize(roads, params_list, ics, xmaxs, cfg.weights,
                            cfg.dt, w_jerk=0.0, w_mag=0.0,
                            n_iter=oracle_iters, lr=oracle_lr,
                            verbose=verbose)
    rows.append(evaluate_controller("PGD upper (privileged)",
                                    make_openloop_cmd(f_pgd, n_steps),
                                    n_steps, **eval_args))
    f_plan = oracle_optimize(roads, params_list, ics, xmaxs, cfg.weights,
                             cfg.dt, w_jerk=cfg.w_jerk, w_mag=cfg.w_mag,
                             n_iter=oracle_iters, lr=oracle_lr,
                             verbose=verbose)
    rows.append(evaluate_controller("planner (privileged)",
                                    make_openloop_cmd(f_plan, n_steps),
                                    n_steps, **eval_args))

    if verbose:
        _print_table(f"mode {cfg.mode} six-metric table "
                     f"(T_w={cfg.t_w}s, B={cfg.B}, eval seed {eval_seed})",
                     rows)
        if cfg.mode == 1:
            print(f"\nsettling_s: "
                  + ", ".join(f"{r['name']}={r['settling_s']:.4f}"
                              for r in rows))
            print("pre/post J_acc: "
                  + ", ".join(f"{r['name']}={r['pre_J_acc']:.4f}/"
                              f"{r['post_J_acc']:.4f}" for r in rows))
        else:
            print(f"\nevent_peak |a2|: "
                  + ", ".join(f"{r['name']}={r['event_peak_acc']:.4f}"
                              for r in rows))
            print("event overshoot (m): "
                  + ", ".join(f"{r['name']}={r['event_overshoot_m']:.4f}"
                              for r in rows))
        pgd = rows[-2]["j_sim"]
        plan = rows[-1]["j_sim"]
        print(f"\noracle gap (planner - PGD upper): {plan - pgd:+.6e} "
              f"(privileged-vs-privileged, logged per T25)")
    return rows, recs


def main():
    ap = argparse.ArgumentParser(description="Plan C policy evaluation")
    ap.add_argument("--mode", type=int, default=2, choices=(1, 2))
    ap.add_argument("--ckpt", type=str, default=None,
                    help="load a trained policy checkpoint (else train)")
    ap.add_argument("--iterations", type=int, default=200,
                    help="training iterations when --ckpt is not given")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--H", type=int, default=100)
    ap.add_argument("--t-w", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=2026)
    ap.add_argument("--oracle-iters", type=int, default=100)
    ap.add_argument("--oracle-lr", type=float, default=5.0e-2)
    ap.add_argument("--fixed-roads", action="store_true",
                    help="train on the fixed T23 road fixture (default: "
                         "per-episode domain randomization, decision 7 -- "
                         "the fixed 8-road fixture overfits and hides the "
                         "preview value)")
    args = ap.parse_args()

    cfg = TrainConfig(mode=args.mode, B=args.batch, H=args.H,
                      t_w=args.t_w, n_iter=args.iterations, seed=args.seed,
                      fixed_road_seed=11 if args.fixed_roads else None)
    policy = None
    if args.ckpt is not None:
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(args.ckpt)
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        policy = PolicyMLP(cfg.H)
        policy.load_state_dict(ck["model"])
        policy.eval()
        print(f"[eval] loaded checkpoint {args.ckpt}")
    run_eval(cfg, policy=policy, eval_seed=args.eval_seed,
             oracle_iters=args.oracle_iters, oracle_lr=args.oracle_lr)


if __name__ == "__main__":
    main()
