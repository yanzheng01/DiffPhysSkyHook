"""Shared Quadrants runtime initialization (single init per process)."""

import quadrants as qd

_initialized = False
_arch = None


def ensure_init(arch=None):
    """Initialize Quadrants exactly once per process.

    Default: x64 compiled backend, float64, adstack enabled (required for
    correct reverse-mode gradients through runtime loops; see
    docs/plan_a_dev_test_plan.md section 2). Pass arch explicitly (e.g.
    qd.cuda) BEFORE importing any skyhook model module to host the fields on
    another backend -- the GPU parity worker (D4) does this in its own
    process. A second call with a different arch is a hard error: fields are
    bound to the first init.
    """
    global _initialized, _arch
    if _initialized:
        if arch is not None and arch is not _arch:
            raise RuntimeError(
                f"Quadrants already initialized on {_arch}; cannot switch to {arch}")
        return
    _arch = qd.x64 if arch is None else arch
    qd.init(arch=_arch, default_fp=qd.f64, ad_stack_experimental_enabled=True)
    _initialized = True
