"""Policy network + observation assembly (plan C, F2 -- docs/plan_c_policy_
training_dev_test_plan.md section 3, decisions 1/6/9/10).

Observation (v1, all channels DETACHED host-side numpy -> torch constants;
privilege declaration: o_c is ground truth by design, decision 9):

  past window   a2 history (accelerometer, 0.5 s) + executed f_cmd history
                (proprioception, 0.5 s), both 10 ms mean-pooled -> 50 pts
                x 2 ch. The (a2, fc) PAIR is the controlled-system
                identification pair (v2.1): without the fc window the
                network cannot tell "road got rougher" from "I just pushed
                harder" -- M1-B/M2-B is not an ablatable channel.
  preview wind. (mode 2 only) (z_evt, zd_evt) EVENT component of the road
                ahead, 0.2 s at 5 ms pooling -> 40 pts x 2 ch. Material
                roughness is INVISIBLE to the forward sensor (v2.3): mode 1
                = mode 2 with the preview channel zeroed (same parameter
                count, the fair-comparison protocol of decision 9).
  current state o_c = (x1, x2, v1, v2, x_max), ground truth (v1 privilege).

Normalization: FIXED scales (decision 9), no runtime statistics:
disp 0.05 m, vel 0.5 m/s, acc 5 m/s^2, force ref = 10 (m2 = 1 gauge),
x_max 0.1 m.

Cold start: the segment-1 past window is zero-padded on the left (P2
convention: quasi-static IC on a nonzero road), host-side rolling buffer.

Architecture (v1, no recurrence -- "remembering" is carried by the window
buffer; GRU ablation deferred to F3): past encoder + preview encoder +
o_c projection -> MLP trunk -> linear head emitting the whole (B, H)
segment block at the segment boundary (10 Hz decision rate).

Init (decision 6): head weights x 0.01 (small), head bias initialized to
the passive-equivalent c_min * v_rel_typ scale so the untrained policy
outputs a small nonzero command (zero-command plateau semantics).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .params import SemiActiveParams

# --- fixed normalization scales (plan C decision 9) ---------------------------
SCALE_DISP = 0.05     # m
SCALE_VEL = 0.5       # m/s
SCALE_ACC = 5.0       # m/s^2
SCALE_FORCE = 10.0    # canonical force unit (m2 = 1 gauge)
SCALE_XMAX = 0.1      # m

# --- window geometry ----------------------------------------------------------
PAST_WIN = 50         # points (0.5 s at 10 ms pooling)
PAST_POOL_MS = 10     # mean-pooling stride for the past windows
PREV_WIN = 40         # points (0.2 s at 5 ms pooling, v2.1)
PREV_POOL_MS = 5      # preview pooling stride (sharper: event shape)


def pool_mean(x: np.ndarray, step_ms: float, dt: float) -> np.ndarray:
    """Mean-pool the last axis of x (1 kHz records) to `step_ms` blocks.

    Block boundaries are aligned to the START of the given series; a
    trailing partial block is pooled with its actual length. Output width
    is ceil(n / blocks). Pure numpy, host-side (anti-aliasing low-pass,
    decision 9 downsampling).
    """
    n = x.shape[-1]
    step = max(1, int(round(step_ms * 1.0e-3 / dt)))
    n_blocks = (n + step - 1) // step
    out = np.empty(x.shape[:-1] + (n_blocks,), dtype=np.float64)
    for j in range(n_blocks):
        lo = j * step
        hi = min(n, lo + step)
        out[..., j] = x[..., lo:hi].mean(axis=-1)
    return out


class ObsBuffer:
    """Rolling past-window buffer (host-side, per batch element).

    Holds the most recent PAST_WIN pooled points of (a2, fc); segment-1
    left side is zero (cold-start convention). push() is called once per
    finished segment with that segment's 1 kHz records.
    """

    def __init__(self, B: int, past_win: int = PAST_WIN):
        self.past_win = past_win
        self.a2 = np.zeros((B, past_win), dtype=np.float64)
        self.fc = np.zeros((B, past_win), dtype=np.float64)

    def push(self, a2_seg: np.ndarray, fc_seg: np.ndarray,
             dt: float) -> None:
        """Append one segment's records (B, H) at 1 kHz, 10 ms pooling."""
        a2p = pool_mean(np.asarray(a2_seg, dtype=np.float64),
                        PAST_POOL_MS, dt)
        fcp = pool_mean(np.asarray(fc_seg, dtype=np.float64),
                        PAST_POOL_MS, dt)
        self.a2 = np.concatenate([self.a2, a2p], axis=1)[:, -self.past_win:]
        self.fc = np.concatenate([self.fc, fcp], axis=1)[:, -self.past_win:]

    def features(self) -> np.ndarray:
        """(B, 2*PAST_WIN) normalized [a2 | fc] past-window features."""
        return np.concatenate([self.a2 / SCALE_ACC, self.fc / SCALE_FORCE],
                              axis=1)


def preview_features(road_seqs, i0: int, dt: float,
                     prev_win: int = PREV_WIN) -> np.ndarray:
    """(B, PREV_WIN, 2) normalized event-component preview window.

    Starting at the CURRENT position (global half-step index i0, the
    segment start), 0.2 s ahead of (z_evt, zd_evt) at 5 ms pooling. The
    forward sensor sees ONLY the event component (v2.3); material
    roughness never enters this channel. Blocks past the end of the road
    grid stay zero (episode tail zero-padding).
    """
    B = len(road_seqs)
    half_ms = dt * 1.0e3 / 2.0
    step = max(1, int(round(PREV_POOL_MS / half_ms)))   # half-steps/block
    out = np.zeros((B, prev_win, 2), dtype=np.float64)
    for b, seq in enumerate(road_seqs):
        for j in range(prev_win):
            lo = i0 + j * step
            hi = min(lo + step, seq.m)
            if lo >= seq.m:
                break                       # episode tail: zero-padded
            out[b, j, 0] = seq.z_evt[lo:hi].mean() / SCALE_DISP
            out[b, j, 1] = seq.zd_evt[lo:hi].mean() / SCALE_VEL
    return out


def oc_features(state: np.ndarray, xmaxs) -> np.ndarray:
    """(B, 5) normalized current-state features (x1, x2, v1, v2, x_max)."""
    state = np.asarray(state, dtype=np.float64)
    xmax = np.asarray(xmaxs, dtype=np.float64)
    return np.stack([state[:, 0] / SCALE_DISP, state[:, 1] / SCALE_DISP,
                     state[:, 2] / SCALE_VEL, state[:, 3] / SCALE_VEL,
                     xmax / SCALE_XMAX], axis=1)


@dataclass
class Obs:
    """One segment-boundary observation (torch constants, detached)."""
    past: torch.Tensor        # (B, 2*PAST_WIN)
    preview: torch.Tensor     # (B, PREV_WIN, 2)  (zeros in mode 1)
    oc: torch.Tensor          # (B, 5)
    v_rel: torch.Tensor       # (B,) raw v2 - v1 (passive bypass input)


def build_obs(buf: ObsBuffer, road_seqs, i0: int, state: np.ndarray,
              xmaxs, dt: float, mode: int, device=None) -> Obs:
    """Assemble the segment-boundary observation (mode 1/2, decision 9).

    device: torch device for the assembled tensors (plan D G3 -- CUDA
    training); None keeps the historical CPU placement.
    """
    B = buf.a2.shape[0]
    state = np.asarray(state, dtype=np.float64)
    past = buf.features()
    if mode == 2:
        prev = preview_features(road_seqs, i0, dt)
    elif mode == 1:
        prev = np.zeros((B, PREV_WIN, 2), dtype=np.float64)
    else:
        raise ValueError(f"mode must be 1 or 2, got {mode}")
    return Obs(past=torch.as_tensor(past, device=device),
               preview=torch.as_tensor(prev, device=device),
               oc=torch.as_tensor(oc_features(state, xmaxs), device=device),
               v_rel=torch.as_tensor(state[:, 3] - state[:, 2],
                                     device=device))


class PolicyMLP(nn.Module):
    """Window-encoder MLP emitting the (B, H) segment command block.

    Mode 1 = mode 2 with a zeroed preview input: the preview encoder
    parameters ALWAYS exist (equal parameter count, decision 9 fair
    protocol); the caller just never feeds a nonzero preview in mode 1.

    Passive bypass (decision 6-iii): the segment block is
        f_cmd = passive_gain * (v2 - v1) + head(...)
    with the head initialized small. A CONSTANT nonzero bias would sit
    outside the force window [c_min*vr, c_max*vr] whenever vr crosses
    zero (the window collapses to {0} there), starving the early sim
    gradients; the passive-equivalent c_min * v_rel rides exactly ON the
    window's lower bound, so the untrained policy is realizable almost
    everywhere (mask ~ 0) and the adjoint flows from step one.
    """

    def __init__(self, H: int, hidden: int = 128):
        super().__init__()
        self.past_enc = nn.Sequential(nn.Linear(2 * PAST_WIN, 64), nn.Tanh())
        self.prev_enc = nn.Sequential(nn.Linear(2 * PREV_WIN, 64), nn.Tanh())
        self.oc_enc = nn.Sequential(nn.Linear(5, 16), nn.Tanh())
        self.trunk = nn.Sequential(
            nn.Linear(64 + 64 + 16, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, H)
        # decision 6: small head weights; the nonzero-command duty is
        # carried by the passive bypass, head bias starts at zero
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)
        self.passive_gain = SemiActiveParams().c_min
        self.double()   # match the bridge / simulation precision (f64)

    def forward(self, obs: Obs) -> torch.Tensor:
        """(B, H) command block for the coming segment (ZOH semantics)."""
        z = torch.cat([
            self.past_enc(obs.past),
            self.prev_enc(obs.preview.reshape(obs.preview.shape[0], -1)),
            self.oc_enc(obs.oc),
        ], dim=1)
        bypass = self.passive_gain * obs.v_rel.unsqueeze(1)
        return bypass + self.head(self.trunk(z))
