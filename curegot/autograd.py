"""Differentiable PyTorch API for the SPLR entropic-OT solver.

Public API (also attached to the native ``curegot.torch`` submodule by
``curegot/__init__.py``):

    import curegot

    T = curegot.torch.splr_ot(M, a, b, eta=0.05)   # like a normal torch op
    loss = downstream_loss(T)
    loss.backward()                                # M.grad = dL/dM

Gradient scope (deliberately conservative):

    M:    differentiable (dL/dM via the validated implicit-differentiation
          path: reduced-Hessian PCG with the native diagonal preconditioner
          by default)
    a, b: constants -- requires_grad=True raises NotImplementedError
    eta:  constant scalar -- tensor requires_grad=True raises NotImplementedError

Requirements: M/a/b are CUDA float64 tensors on the same device; one OT
problem [n, m] per call, with n >= 1 and m >= 2.
AMP / autocast is unsupported and raises. Double backward is unsupported
(@once_differentiable raises on the second differentiation).

Non-convergence policy (production-safe): forward non-convergence and
backward PCG non-convergence both raise RuntimeError rather than returning
silently bad values. The lower-level manual API
(curegot.torch.sinkhorn_splr_save_ctx + curegot.torch.solve_backward) keeps its
flag-returning semantics for research use.

This module changes NO solver math: forward is the native
sinkhorn_splr_save_ctx, backward is curegot.backward.solve_backward. The
default backward preconditioner is the native diagonal preconditioner.
"""

import math

import torch
from torch.autograd.function import once_differentiable

from ._internal_torch import torch as _native_torch
from .backward import solve_backward, _VALID_PRECONDITIONERS

__all__ = [
    "splr_ot",
    "SPLROTFunction",
]


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------

def _check_inputs(M, a, b):
    """Enforce the device/dtype/shape contract.

    Strict on purpose: the native wrapper would silently convert
    float32/CPU inputs to float64 CUDA and convert outputs back, which
    would mislead users about the gradient dtype/device. We raise instead.
    """
    for name, t in (("M", M), ("a", a), ("b", b)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"splr_ot: {name} must be a torch.Tensor, "
                            f"got {type(t).__name__}")
        if not t.is_cuda:
            raise RuntimeError(
                f"splr_ot: {name} must be a CUDA tensor "
                f"(splr_ot supports CUDA float64 only)"
            )
        if t.dtype != torch.float64:
            raise RuntimeError(
                f"splr_ot: {name} must be float64, got {t.dtype}; "
                "the native solver is float64-only and silent conversion "
                "would make the gradient dtype misleading"
            )
    if M.dim() != 2:
        raise ValueError(
            f"splr_ot: M must be 2D; one OT problem is supported per call, "
            f"got shape {tuple(M.shape)}"
        )
    n, m = M.shape
    if a.dim() != 1 or a.shape[0] != n:
        raise ValueError(
            f"splr_ot: a must be 1D of length {n}, got shape {tuple(a.shape)}"
        )
    if b.dim() != 1 or b.shape[0] != m:
        raise ValueError(
            f"splr_ot: b must be 1D of length {m}, got shape {tuple(b.shape)}"
        )
    if not (M.device == a.device == b.device):
        raise ValueError(
            f"splr_ot: M/a/b must be on the same device, got "
            f"{M.device}, {a.device}, {b.device}"
        )


def _check_eta_gradient(eta):
    """Reject differentiable eta before scalar conversion can detach it."""
    if isinstance(eta, torch.Tensor) and eta.requires_grad:
        raise NotImplementedError(
            "splr_ot: gradients w.r.t. eta are not supported; "
            "eta must be a constant scalar"
        )


def _validate_scalars(eta, tol, max_iter, density, candidate_sinkhorn_iter,
                      backward_tol, backward_max_iter):
    """Lightweight scalar-parameter validation for the public API.

    Catches obviously invalid values with clear errors before they reach
    the native layer. Deliberately minimal (no over-defensive checks).
    """
    _check_eta_gradient(eta)
    if not (math.isfinite(eta) and eta > 0):
        raise ValueError(f"splr_ot: eta must be a positive finite scalar, "
                         f"got {eta}")
    if not (math.isfinite(tol) and tol > 0):
        raise ValueError(f"splr_ot: tol must be positive, got {tol}")
    if not int(max_iter) > 0:
        raise ValueError(f"splr_ot: max_iter must be positive, "
                         f"got {max_iter}")
    if density is not None and not (0.0 < float(density) <= 1.0):
        raise ValueError(f"splr_ot: density must be in (0, 1], "
                         f"got {density}")
    if not int(candidate_sinkhorn_iter) >= 0:
        raise ValueError(f"splr_ot: candidate_sinkhorn_iter must be "
                         f">= 0, got {candidate_sinkhorn_iter}")
    if not (math.isfinite(backward_tol) and backward_tol > 0):
        raise ValueError(f"splr_ot: backward_tol must be positive, "
                         f"got {backward_tol}")
    if not int(backward_max_iter) > 0:
        raise ValueError(f"splr_ot: backward_max_iter must be positive, "
                         f"got {backward_max_iter}")


# ---------------------------------------------------------------------------
# autograd.Function
# ---------------------------------------------------------------------------

class SPLROTFunction(torch.autograd.Function):
    """Custom autograd Function wrapping forward + implicit backward.

    forward runs the native SPLR solver and stashes the plan and the
    pybind backward context on the autograd ctx; backward runs the shared
    solve_backward (native C++ PCG) with the chosen
    preconditioner.
    """

    @staticmethod
    def forward(ctx, M, a, b, eta, tol, max_iter, density,
                candidate_sinkhorn_iter, backward_preconditioner,
                backward_tol, backward_max_iter):
        kwargs = {
            "candidate_sinkhorn_iter": candidate_sinkhorn_iter,
        }
        if density is not None:
            kwargs["density"] = density

        # Run forward and retain its final plan and preconditioner context.
        out = _native_torch.sinkhorn_splr_save_ctx(
            M, a, b, eta, tol, max_iter, 0, **kwargs
        )
        if not bool(out["converged"]):
            # Production-safe default: never hand a non-converged plan to a
            # training loop.
            raise RuntimeError(
                f"splr_ot: forward solver did not converge "
                f"(niter={int(out['niter'])}, max_iter={max_iter}, "
                f"tol={tol}); relax tol or increase max_iter"
            )
        T = out["plan"]

        # The graph owns the saved tensor and the separate pybind context.
        ctx.save_for_backward(T)
        ctx.backward_context = out["context"]
        ctx.eta = eta
        ctx.backward_preconditioner = backward_preconditioner
        ctx.backward_tol = backward_tol
        ctx.backward_max_iter = backward_max_iter

        return T

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (T,) = ctx.saved_tensors

        # Make strided upstream gradients contiguous at the boundary.
        G = grad_output if grad_output.is_contiguous() \
            else grad_output.contiguous()

        # Build the RHS, dispatch native PCG and assemble the cost gradient.
        res = solve_backward(
            T, G, ctx.backward_context, ctx.eta,
            preconditioner=ctx.backward_preconditioner,
            tol=ctx.backward_tol,
            max_iter=ctx.backward_max_iter,
        )
        if not res["converged"]:
            # Production-safe default: never return a silently unconverged
            # gradient into a training loop.
            raise RuntimeError(
                f"splr_ot: backward PCG did not converge "
                f"(iterations={res['iterations']}, "
                f"max_iter={ctx.backward_max_iter}, "
                f"tol={ctx.backward_tol}, "
                f"preconditioner='{ctx.backward_preconditioner}')"
            )
        grad_M = res["grad"]

        # One gradient per forward input (M, a, b, eta, tol, max_iter,
        # density, candidate_sinkhorn_iter, backward_preconditioner,
        # backward_tol, backward_max_iter) -- order must match exactly.
        return (grad_M, None, None, None, None, None, None, None, None,
                None, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def splr_ot(
    M,
    a,
    b,
    *,
    eta,
    tol=1e-6,
    max_iter=1000,
    density=None,
    candidate_sinkhorn_iter=3,
    backward_preconditioner="diag",
    backward_tol=1e-6,
    backward_max_iter=2000,
):
    """Differentiable entropic-OT plan via the SPLR solver.

    Args:
        M:   [n, m] CUDA float64 tensor (cost matrix); the only
             differentiable input. One OT problem per call.
        a:   [n] CUDA float64 source marginal (constant).
        b:   [m] CUDA float64 target marginal (constant).
        eta: entropic regularization (constant scalar, required, > 0).
        tol: forward solver tolerance. Default 1e-6 matches the native
             pybind default of sinkhorn_splr_save_ctx.
        max_iter: forward iteration cap (native default 1000).
        density: sparsification budget in (0, 1]; None -> native default
             min(10 / min(n, m), 1.0).
        candidate_sinkhorn_iter: native default 3.
        backward_preconditioner: "diag" (default, native diagonal) |
             "splr" (cuDSS + rank-2) | "none" (plain CG, debug).
        backward_tol: backward PCG relative residual tolerance. Default: 1e-6.
        backward_max_iter: backward PCG cap (2000).

    Returns:
        T: [n, m] CUDA float64 tensor, the optimal transport plan, with
           grad_fn attached when M.requires_grad.

    Raises:
        RuntimeError: CUDA/float64 contract violation, AMP/autocast active,
             forward non-convergence, backward PCG non-convergence.
        NotImplementedError: a, b or tensor eta requires grad.
        ValueError: invalid preconditioner, shape mismatch, or invalid
             scalar parameter.

    Example:
        >>> import curegot
        >>> M = M.cuda().double().requires_grad_(True)
        >>> T = curegot.torch.splr_ot(M, a, b, eta=0.05)
        >>> loss = downstream_loss(T)
        >>> loss.backward()   # M.grad holds dL/dM

    Notes:
        * Only dL/dM is supported; a/b/eta are constants.
        * CUDA float64 only; AMP/autocast is rejected.
        * Double backward is unsupported (@once_differentiable).
        * Forward and backward non-convergence raise RuntimeError.
        * Adding a constant c to all entries of M leaves the plan
          unchanged (absorbed into the scalings).
    """
    if torch.is_autocast_enabled():
        raise RuntimeError(
            "splr_ot: AMP/autocast is unsupported (the solver is "
            "float64-only); disable autocast for this call"
        )
    _check_inputs(M, a, b)
    _validate_scalars(eta, tol, max_iter, density, candidate_sinkhorn_iter,
                      backward_tol, backward_max_iter)
    if a.requires_grad or b.requires_grad:
        raise NotImplementedError(
            "splr_ot: gradients w.r.t. a/b are not supported "
            "(only dL/dM); detach them or use the lower-level API"
        )
    if backward_preconditioner not in _VALID_PRECONDITIONERS:
        raise ValueError(
            f"splr_ot: invalid backward_preconditioner "
            f"'{backward_preconditioner}'; expected one of: "
            f"{' | '.join(_VALID_PRECONDITIONERS)}"
        )
    return SPLROTFunction.apply(
        M, a, b,
        float(eta), float(tol), int(max_iter), density,
        int(candidate_sinkhorn_iter), backward_preconditioner,
        float(backward_tol), int(backward_max_iter),
    )
