"""D0 toolchain smoke tests (Gate: compiled-backend hello-ODE + adstack gradient check).

Stage D0 of docs/plan_a_dev_test_plan.md:
  1. hello-ODE: harmonic oscillator integrated by RK4 on the compiled (x64/LLVM)
     backend. Time-stepping loop stays in the Python host layer (architecture
     rule); each step is one @qd.kernel call composed with a @qd.func RHS.
     Validated against the analytical solution x(t) = x0*cos(w*t) + v0/w*sin(w*t).
  2. adstack gradient check: reverse-mode autodiff via qd.ad.Tape with
     ad_stack_experimental_enabled=True, against known closed-form gradients:
       - scalar:  y = x**2          -> dy/dx = 2x
       - dynamic loop: y = sum(x_i) -> dy/dx_i = 1  (loop accumulation path,
         the known silent-wrong-gradient risk area when adstack is off)

All real arithmetic in float64. Prints in English only.
"""

import math

import pytest
import quadrants as qd

from skyhook._runtime import ensure_init

ensure_init()

# ---------------------------------------------------------------------------
# Part 1: hello-ODE (harmonic oscillator, RK4, compiled backend)
# ---------------------------------------------------------------------------
# State fields (0-d scalars): position and velocity.
pos = qd.field(dtype=qd.f64, shape=())
vel = qd.field(dtype=qd.f64, shape=())


@qd.func
def accel(x: qd.f64, omega: qd.f64) -> qd.f64:
    """RHS of the harmonic oscillator: dv/dt = -omega^2 * x."""
    return -omega * omega * x


@qd.kernel
def rk4_step(omega: qd.f64, dt: qd.f64):
    """Advance the 0-d state fields by one RK4 step of the harmonic oscillator."""
    x0 = pos[None]
    v0 = vel[None]

    # Stage 1
    a1 = accel(x0, omega)
    # Stage 2
    x2 = x0 + 0.5 * dt * v0
    v2 = v0 + 0.5 * dt * a1
    a2 = accel(x2, omega)
    # Stage 3
    x3 = x0 + 0.5 * dt * v2
    v3 = v0 + 0.5 * dt * a2
    a3 = accel(x3, omega)
    # Stage 4
    x4 = x0 + dt * v3
    v4 = v0 + dt * a3
    a4 = accel(x4, omega)

    # Combine
    pos[None] = x0 + (dt / 6.0) * (v0 + 2.0 * v2 + 2.0 * v3 + v4)
    vel[None] = v0 + (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)


def run_hello_ode(x0: float, v0: float, omega: float, dt: float, n_steps: int):
    """Python host layer owns the time loop; one kernel launch per step."""
    pos[None] = x0
    vel[None] = v0
    for _ in range(n_steps):
        rk4_step(omega, dt)
    return pos[None], vel[None]


class TestHelloODE:
    """Gate 1: compiled-backend hello-ODE runs and matches analytics."""

    def test_harmonic_oscillator_rk4_vs_analytic(self):
        x0, v0 = 1.0, 0.0
        omega = 2.0 * math.pi  # 1 Hz
        dt = 1.0e-3  # locked project timestep
        n_steps = 2000  # 2 s

        x_end, v_end = run_hello_ode(x0, v0, omega, dt, n_steps)
        t_end = n_steps * dt

        x_exact = x0 * math.cos(omega * t_end) + (v0 / omega) * math.sin(omega * t_end)
        v_exact = -x0 * omega * math.sin(omega * t_end) + v0 * math.cos(omega * t_end)

        # Tolerances: genuine RK4 global truncation error at dt=1 ms over 2 s
        # is ~1e-9 for velocity; the toolchain check needs 1e-8, integrator
        # convergence order itself is verified later in D4/T10.
        print(f"[D0] hello-ODE RK4 vs analytic: x_err={abs(x_end - x_exact):.3e}, "
              f"v_err={abs(v_end - v_exact):.3e}")
        assert abs(x_end - x_exact) < 1e-8
        assert abs(v_end - v_exact) < 1e-8


# ---------------------------------------------------------------------------
# Part 2: adstack reverse-mode gradient check against closed forms
# ---------------------------------------------------------------------------
# Scalar case: y = x**2, dy/dx = 2x.
x_s = qd.field(dtype=qd.f64, shape=(), needs_grad=True)
y_s = qd.field(dtype=qd.f64, shape=(), needs_grad=True)


@qd.kernel
def square():
    y_s[None] = x_s[None] * x_s[None]


# Dynamic-loop case: y = sum_i x_i, dy/dx_i = 1 for all i.
# The runtime-bounded range() loop is exactly the construct whose reverse
# gradient silently goes wrong when the adstack extension is disabled.
# Reverse-mode global data access rule: a kernel mixing loops and loop-free
# global stores is rejected, so accumulation must live inside the loop and
# the clear lives in a separate loop-free kernel (split into two kernels).
N_ARR = 16
x_a = qd.field(dtype=qd.f64, shape=N_ARR, needs_grad=True)
y_a = qd.field(dtype=qd.f64, shape=(), needs_grad=True)


@qd.kernel
def clear_sum():
    y_a[None] = 0.0


@qd.kernel
def sum_dynamic(n: qd.i32):
    for i in range(n):
        y_a[None] += x_a[i]


class TestAdstackGradient:
    """Gate 2: adstack reverse-mode gradients match known closed forms."""

    def test_adstack_enabled(self):
        assert qd.is_extension_enabled(qd.extension.adstack), (
            "adstack extension must be enabled for reverse-mode gradients"
        )

    def test_scalar_square_gradient(self):
        for xv in (0.5, 3.0, -1.25):
            x_s[None] = xv
            with qd.ad.Tape(loss=y_s):
                square()
            grad = x_s.grad[None]
            expect = 2.0 * xv
            print(f"[D0] adstack scalar y=x^2 at x={xv}: grad={grad}, expect={expect}")
            assert abs(grad - expect) < 1e-12 * max(1.0, abs(expect))

    def test_dynamic_loop_sum_gradient(self):
        n = N_ARR
        vals = [float(i + 1) for i in range(n)]
        for i in range(n):
            x_a[i] = vals[i]
        clear_sum()
        with qd.ad.Tape(loss=y_a):
            sum_dynamic(n)
        grads = [x_a.grad[i] for i in range(n)]
        expect = [1.0] * n
        max_err = max(abs(g - e) for g, e in zip(grads, expect))
        print(f"[D0] adstack dynamic-loop sum grad: max_err={max_err:.3e} (expect all 1.0)")
        assert max_err < 1e-12


if __name__ == "__main__":
    sys_args = ["-v", "-s", __file__]
    raise SystemExit(pytest.main(sys_args))
