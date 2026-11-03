"""torch <-> Quadrants gradient bridge (plan C, F1 -- docs/plan_c_policy_
training_dev_test_plan.md section 3, decision 2).

Segment-level SimLoss: a torch.autograd.Function wrapping the F0 quadrants
stack (model3 + sim3). The pattern follows DiffPhysDrone's RunFunction:

  forward : host-side setup (_setup / _setup_road / _setup_fcmd, all
            OUTSIDE any Tape/FwdMode -- a store inside the context would
            sever the adjoint chain, D3 pattern) -> one Tape-wrapped
            run_jsim_chain -> the scalar j_mean3. The (B, H) adjoint
            dJ_sim_mean/dfc3 is read from fc3.grad right after the Tape
            exits and COPIED (the next forward overwrites it).
  backward: the stored adjoint, optionally weighted by the within-segment
            time decay lambda^n (plan C decision 6-ii; default 1.0 --
            segment-boundary truncation already caps the credit horizon,
            lambda is a fine-tuning knob), scaled by the incoming seed.

Zero new math: the rollout, the force clip and the J reduction are the
Plan A/B-validated quadrants kernels; the bridge only moves values and
adjoints across the torch boundary.

F1 spike verdicts baked into the design (2026-11-02):
  - torch CUDA and Quadrants coexist in one process (interleaved CUDA
    matmul + Tape runs give identical J), so f_cmd may live on any device;
    the data path itself goes through CPU numpy (host-side writes).
  - element-wise fc writes are fast enough at steady state (~18 ms per
    segment including the Tape; the 2.3 s first call is JIT compilation),
    so no tensor-copy write path is needed.

Discipline (parameters_reference.md section 4.4): training passes NONZERO
ICs -- x0s is therefore a REQUIRED argument (unlike sim3.run_cmd_seq_batch
there is no rest-IC default here to reach for by accident).
"""

import numpy as np
import torch
import quadrants as qd

from . import model3, sim3


# --- road-write cache (F2: the K segment calls of one episode reuse the ---
# --- same RoadSeq objects; rewriting 2*m*B half-step samples per segment  ---
# --- dominated the bridge cost, T22)                                       ---
# The cache holds STRONG references to the cached sequences, so the ids in
# the key cannot be recycled by the garbage collector. Discipline: never
# mutate a RoadSeq in place after passing it to sim_loss (regenerate a new
# object instead) -- the generators in road_seq.py always build fresh ones.
_road_cache = {"key": None, "refs": None}


def _setup_road_cached(road_seqs, i0: int, n_steps: int, B: int) -> None:
    """_setup_road with an identity-keyed write cache.

    Validation (shape / coverage) runs on every call; only the element-wise
    host writes are skipped when the sequence objects are unchanged.
    """
    road_seqs = list(road_seqs)
    key = tuple(id(s) for s in road_seqs)
    if _road_cache["key"] != key:
        sim3._setup_road(road_seqs, i0, n_steps, B)
        _road_cache["key"] = key
        _road_cache["refs"] = road_seqs
        return
    # cache hit: re-validate coverage only (fields are already on device)
    if len(road_seqs) != B:
        raise ValueError(f"len(road_seqs)={len(road_seqs)} != B={B}")
    if i0 < 0 or i0 % 2 != 0:
        raise ValueError(f"i0 must be a non-negative even half-step index, "
                         f"got {i0}")
    for b in range(B):
        seq = road_seqs[b]
        if not isinstance(seq, sim3.RoadSeq):
            raise ValueError(f"road_seqs[{b}] is not a RoadSeq")
        if i0 + 2 * n_steps > seq.m - 1:
            raise ValueError(
                f"road_seqs[{b}] covers {seq.m} half-steps; rollout needs "
                f"i0+2*n_steps+1 = {i0 + 2 * n_steps + 1}")


class SegmentRecords:
    """Whole-step records of the LAST bridge forward (numpy copies).

    Read via read_last_segment() BEFORE the next sim_loss call (the next
    rollout overwrites the model3 fields). final_state is the segment-end
    state (B, 4) the training loop hands to the next segment (detached by
    construction -- plain numpy, plan C decision 1).
    """

    def __init__(self, n_steps: int, B: int):
        self.state = model3.traj3.to_numpy()[:B, :n_steps].copy()
        self.a2 = model3.acc3.to_numpy()[:B, :n_steps].copy()
        self.fd = model3.fd3.to_numpy()[:B, :n_steps].copy()
        self.mask = model3.mask3.to_numpy()[:B, :n_steps].copy()
        self.travel = model3.travel3.to_numpy()[:B, :n_steps].copy()
        self.final_state = model3.xh3.to_numpy()[:B, n_steps].copy()
        # pure bridge adjoint dJ_sim/df_cmd of the last forward (BEFORE the
        # torch-side regularizers mix into f.grad) -- the F2 adjoint
        # diagnostic (T23 "adjoint nonzero fraction")
        self.adjoint = model3.fc3.grad.to_numpy()[:B, :n_steps].copy()


def read_last_segment(n_steps: int, B: int) -> SegmentRecords:
    """Snapshot of the last segment's records (see SegmentRecords)."""
    return SegmentRecords(n_steps, B)


class _SimLossFunction(torch.autograd.Function):
    """autograd.Function core; use the sim_loss() wrapper, not this."""

    @staticmethod
    def forward(ctx, f_cmd, road_seqs, params_list, x0s, xmaxs, weights,
                dt, i0, decay):
        B, n_steps = f_cmd.shape
        fc = f_cmd.detach().cpu().numpy().astype(np.float64, copy=True)

        # setup strictly outside the Tape (D3 pattern)
        sim3._setup(params_list, x0s, xmaxs)
        _setup_road_cached(road_seqs, i0, n_steps, B)
        sim3._setup_fcmd(fc, n_steps, B)

        with qd.ad.Tape(loss=model3.j_mean3):
            sim3.run_jsim_chain(n_steps, dt, B, i0, weights)
        j = float(model3.j_mean3[None])
        adj = model3.fc3.grad.to_numpy()[:B, :n_steps].copy()

        ctx._adj = adj          # plain ndarray: survives the next forward
        ctx._decay = float(decay)
        return f_cmd.new_tensor(j)

    @staticmethod
    def backward(ctx, grad_out):
        adj = ctx._adj
        if ctx._decay != 1.0:
            w = ctx._decay ** np.arange(adj.shape[1], dtype=np.float64)
            adj = adj * w[np.newaxis, :]
        g = torch.from_numpy(np.ascontiguousarray(adj))
        g = g.to(device=grad_out.device, dtype=grad_out.dtype)
        return g * grad_out, None, None, None, None, None, None, None, None


def sim_loss(f_cmd, road_seqs, params_list, x0s, xmaxs=None,
             weights=sim3.DEFAULT_WEIGHTS, dt=1.0e-3, i0: int = 0,
             decay: float = 1.0) -> torch.Tensor:
    """Segment-level differentiable sim loss J_sim(f_cmd).

    f_cmd    : (B, H) torch tensor (any device; H = segment steps, ZOH).
               Requires grad from the caller (e.g. a policy output).
    road_seqs: B RoadSeq covering the WHOLE episode half-step grid; the
               segment reads it with the global offset i0.
    x0s      : B initial states (x1, x2, v1, v2) of THIS segment -- the
               training loop passes the previous segment's detached final
               state (plan C decision 1: segment-boundary detach).
    xmaxs    : per-batch stroke envelopes (default X_MAX_DEFAULT each).
    weights  : (w_acc, w_disp, w_tire, w_lim) composite-loss weights.
    i0       : global half-step index of the segment start (even).
    decay    : within-segment adjoint decay lambda^n, n = 0..H-1
               (default 1.0; plan C decision 6-ii).

    Returns the scalar J_sim as a torch tensor connected to f_cmd's graph.
    """
    if f_cmd.dim() != 2:
        raise ValueError(f"f_cmd must be (B, H), got shape {tuple(f_cmd.shape)}")
    if not (0.0 < decay <= 1.0):
        raise ValueError(f"decay must be in (0, 1], got {decay}")
    if x0s is None:
        raise ValueError("x0s is required (nonzero-IC discipline, "
                         "parameters_reference.md section 4.4)")
    if xmaxs is None:
        xmaxs = [sim3.X_MAX_DEFAULT] * f_cmd.shape[0]
    return _SimLossFunction.apply(f_cmd, road_seqs, params_list, x0s, xmaxs,
                                  tuple(weights), float(dt), int(i0), decay)
