"""Optimization loops over the semi-active quarter-car (plan A D5 + plan B E3).

optimize (plan A, Gate T12): projected gradient descent on the physical
parameters theta = (gamma_sky, c_min, c_max) with the constraint set
    gamma_sky >= 0,  0 <= c_min < c_max
projected after every step. Gradients come from the DEFAULT FORWARD JVP
route (plan D5): one qd.ad.FwdMode sweep per tunable parameter over the
single-kernel differentiable rollout (model2.rollout_semi + reduce_j2 +
finalize_j2, the exact launch sequence validated three-way by T13).

optimize_force (plan B, Gate T17): gradient descent on the (n_steps,) force
command series f_cmd driving rollout_cmd_semi. NO optimizer-side projection
(plan B decision 2): realizability is enforced model-side by fd_clip_cmd, so
the iterate space is all of R^T and the clip truncation gives the subgradient
semantics. Gradients come from the DEFAULT REVERSE TAPE route (plan B
decision 3): one backward pass yields the whole (T,) adjoint (three-way
validated by T16).

Nonsmoothness: J is piecewise smooth in both theta and f_cmd (the clip kinks
move with the iterate), so the Armijo backtracking line search accepts only
steps with sufficient decrease; an exhausted line search stops the loop
loudly (stopped = "line_search") instead of creeping along a kink.

Record semantics: iterate k = index k (k = 0 is the start) and
grads[k] = dJ/d(iterate) at iterate k; the gradient of the final iterate is
left as NaN when the loop ends right after an accepted step.
"""

from dataclasses import dataclass

import numpy as np
import quadrants as qd

from . import model2
from .params import SemiActiveParams

PARAM_FIELDS = {"gamma_sky": model2.p_gsky2,
                "c_min": model2.p_cmin2,
                "c_max": model2.p_cmax2}
PARAM_ORDER = ("gamma_sky", "c_min", "c_max")


@dataclass
class OptResult:
    """Iterate record of one optimize() run."""

    th: np.ndarray        # (K+1, 3) iterates (gamma_sky, c_min, c_max)
    j: np.ndarray         # (K+1,) J_acc at each iterate
    grads: np.ndarray     # (K+1, 3) JVP gradient at each iterate (NaN = not computed)
    alphas: np.ndarray     # (K,) accepted step sizes (one per accepted step)
    n_evals: int          # total forward rollouts spent
    stopped: str          # "max_iter" | "line_search" | "zero_gradient"

    @property
    def n_steps(self) -> int:
        return len(self.j) - 1


def _theta_of(p: SemiActiveParams) -> np.ndarray:
    return np.array([p.gamma_sky, p.c_min, p.c_max])


def _params_of(p: SemiActiveParams, th: np.ndarray) -> SemiActiveParams:
    return SemiActiveParams(mu=p.mu, omega_t_sq=p.omega_t_sq,
                            omega_s_sq=p.omega_s_sq, gamma_t=p.gamma_t,
                            gamma_sky=float(th[0]), c_min=float(th[1]),
                            c_max=float(th[2]))


def _project(th: np.ndarray, c_gap: float) -> np.ndarray:
    """Project onto {gamma_sky >= 0, 0 <= c_min, c_max >= c_min + c_gap}."""
    out = th.copy()
    out[0] = max(out[0], 0.0)
    out[1] = max(out[1], 0.0)
    out[2] = max(out[2], out[1] + c_gap)
    return out


def rollout_j(params: SemiActiveParams, road, x0, n_steps: int,
              dt: float) -> float:
    """Plain forward J_acc (no AD context); the line-search workhorse."""
    model2.set_params2(params.mu, params.omega_t_sq, params.omega_s_sq,
                      params.gamma_t, params.gamma_sky, params.c_min,
                      params.c_max)
    model2.set_state_hist2(*x0)
    model2.clear_j2()
    model2.rollout_semi(n_steps, dt, *road.kernel_args())
    model2.reduce_j2(n_steps)
    model2.finalize_j2(n_steps)
    return model2.j_val2[None]


def rollout_grad(params: SemiActiveParams, road, x0, n_steps: int, dt: float,
                 which=PARAM_ORDER):
    """J_acc and its JVP gradients along `which` (forward-mode default route).

    One FwdMode sweep per parameter (the interface supports a single param
    group per sweep); frozen slots come back as exact zeros. Setup stays
    outside every FwdMode context (a parameter store inside would sever the
    dual chain, D3 pattern).
    """
    j = rollout_j(params, road, x0, n_steps, dt)
    grad = np.zeros(3)
    for k, name in enumerate(PARAM_ORDER):
        if name not in which:
            continue
        model2.set_state_hist2(*x0)
        model2.clear_j2()
        with qd.ad.FwdMode(loss=model2.j_val2, param=PARAM_FIELDS[name]):
            model2.rollout_semi(n_steps, dt, *road.kernel_args())
            model2.reduce_j2(n_steps)
            model2.finalize_j2(n_steps)
        grad[k] = model2.j_val2.dual[None]
    return j, grad


def optimize(params: SemiActiveParams, road, x0=(0.0, 0.0, 0.0, 0.0),
             n_steps: int = 1000, dt: float = 1.0e-3, which=PARAM_ORDER,
             n_iters: int = 20, alpha0: float = 1.0, c1: float = 1.0e-4,
             shrink: float = 0.5, alpha_min: float = 1.0e-6,
             c_gap: float = 0.05) -> OptResult:
    """Projected gradient descent with Armijo backtracking (Gate T12).

    which: subset of ("gamma_sky", "c_min", "c_max") to optimize (D5: the
    single-variable gamma_sky stage first, then the full trio).
    """
    theta = _theta_of(params)
    j_cur = rollout_j(params, road, x0, n_steps, dt)
    n_evals = 1
    ths = [theta]
    js = [j_cur]
    gs = [np.full(3, np.nan)]
    alphas = []
    stopped = "max_iter"

    for _ in range(n_iters):
        _, g = rollout_grad(params, road, x0, n_steps, dt, which)
        n_evals += 1 + len([n for n in PARAM_ORDER if n in which])
        gs[-1] = g
        g_norm_sq = float(np.dot(g, g))
        if g_norm_sq == 0.0:
            stopped = "zero_gradient"
            break

        alpha = alpha0
        accepted = False
        while alpha >= alpha_min:
            th_new = _project(theta + alpha * (-g), c_gap)
            p_new = _params_of(params, th_new)
            j_new = rollout_j(p_new, road, x0, n_steps, dt)
            n_evals += 1
            # Armijo: sufficient decrease along the (projected) direction
            if j_new <= j_cur - c1 * alpha * g_norm_sq:
                theta, j_cur, params = th_new, j_new, p_new
                accepted = True
                break
            alpha *= shrink
        if not accepted:
            stopped = "line_search"
            break
        alphas.append(alpha)
        ths.append(theta)
        js.append(j_cur)
        gs.append(np.full(3, np.nan))

    return OptResult(th=np.vstack(ths), j=np.array(js),
                     grads=np.vstack(gs), alphas=np.array(alphas),
                     n_evals=n_evals, stopped=stopped)


# ---------------------------------------------------------------------------
# Plan B (E3): open-loop force-sequence optimization (Gate T17)
# ---------------------------------------------------------------------------
@dataclass
class ForceOptResult:
    """Iterate record of one optimize_force() run."""

    fc: np.ndarray        # (K+1, n_steps) command iterates
    j: np.ndarray         # (K+1,) J_acc at each iterate
    grads: np.ndarray     # (K+1, n_steps) adjoint at each iterate (NaN = not computed)
    alphas: np.ndarray    # (K,) accepted step sizes (one per accepted step)
    n_evals: int          # total forward rollouts spent (backward ≈ 1 rollout)
    stopped: str          # "max_iter" | "line_search" | "zero_gradient"

    @property
    def n_iters(self) -> int:
        return len(self.j) - 1


def _setup_cmd(params: SemiActiveParams, f_cmd: np.ndarray,
               x0, road, n_steps: int):
    """Launch sequence prefix for the command-driven chain (setup outside
    every AD context -- the D3/D5 pattern)."""
    model2.set_params2(params.mu, params.omega_t_sq, params.omega_s_sq,
                       params.gamma_t, params.gamma_sky, params.c_min,
                       params.c_max)
    model2.set_state_hist2(*x0)
    for n in range(n_steps):
        model2.set_fcmd_elem(n, float(f_cmd[n]))
    model2.clear_j2()


def _j_chain(n_steps: int, dt: float, road_args):
    """Differentiable J chain for the command-driven rollout."""
    model2.rollout_cmd_semi(n_steps, dt, *road_args)
    model2.reduce_j2(n_steps)
    model2.finalize_j2(n_steps)


def rollout_force_j(f_cmd: np.ndarray, params: SemiActiveParams, road, x0,
                    n_steps: int, dt: float) -> float:
    """Plain forward J_acc of the command-driven rollout (line-search workhorse)."""
    _setup_cmd(params, f_cmd, x0, road, n_steps)
    _j_chain(n_steps, dt, road.kernel_args())
    return model2.j_val2[None]


def force_adjoint(f_cmd: np.ndarray, params: SemiActiveParams, road, x0,
                  n_steps: int, dt: float) -> tuple:
    """J_acc and the whole (n_steps,) adjoint dJ/df_cmd from ONE reverse pass."""
    _setup_cmd(params, f_cmd, x0, road, n_steps)
    with qd.ad.Tape(loss=model2.j_val2):
        _j_chain(n_steps, dt, road.kernel_args())
    return model2.j_val2[None], model2.f_cmd.grad.to_numpy()[:n_steps].copy()


def optimize_force(params: SemiActiveParams, road, x0=(0.0, 0.0, 0.0, 0.0),
                   n_steps: int = 1000, dt: float = 1.0e-3, f_cmd0=None,
                   n_iters: int = 20, alpha0: float = 10.0, c1: float = 1.0e-4,
                   shrink: float = 0.5, alpha_min: float = 1.0e-8) -> ForceOptResult:
    """Gradient descent on the open-loop force series with Armijo backtracking.

    Start f_cmd0 (default zeros = the passive c_min damper, T14 anchor).
    NO projection (plan B decision 2): the model-side clip makes every
    command realizable; the step direction is the raw (possibly truncated-
    subgradient) Tape adjoint. stopped semantics mirror optimize().
    """
    if f_cmd0 is None:
        fc = np.zeros(n_steps)
    else:
        fc = np.array(f_cmd0, dtype=np.float64)
        if fc.shape != (n_steps,):
            raise ValueError(f"f_cmd0 must have shape ({n_steps},), got {fc.shape}")

    j_cur = rollout_force_j(fc, params, road, x0, n_steps, dt)
    n_evals = 1
    fcs = [fc.copy()]
    js = [j_cur]
    gs = [np.full(n_steps, np.nan)]
    alphas = []
    stopped = "max_iter"

    for _ in range(n_iters):
        _, g = force_adjoint(fc, params, road, x0, n_steps, dt)
        n_evals += 1
        gs[-1] = g
        g_norm_sq = float(np.dot(g, g))
        if g_norm_sq == 0.0:
            stopped = "zero_gradient"
            break

        alpha = alpha0
        accepted = False
        while alpha >= alpha_min:
            fc_new = fc - alpha * g          # no projection (plan B section 2-2)
            j_new = rollout_force_j(fc_new, params, road, x0, n_steps, dt)
            n_evals += 1
            if j_new <= j_cur - c1 * alpha * g_norm_sq:
                fc, j_cur = fc_new, j_new
                accepted = True
                break
            alpha *= shrink
        if not accepted:
            stopped = "line_search"
            break
        alphas.append(alpha)
        fcs.append(fc.copy())
        js.append(j_cur)
        gs.append(np.full(n_steps, np.nan))

    return ForceOptResult(fc=np.vstack(fcs), j=np.array(js),
                          grads=np.vstack(gs), alphas=np.array(alphas),
                          n_evals=n_evals, stopped=stopped)
