#include "backward_solver.h"
#include "backward_context.h"
#include "utils.h"
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Native backward CUDA: ") + cudaGetErrorString(status));
}

// Own the scratch buffers until all caller-stream work has completed.
struct Workspace
{
    cudaStream_t stream;
    double* device = nullptr;
    double* pinned = nullptr;

    explicit Workspace(cudaStream_t s) : stream(s) {}
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    ~Workspace()
    {
        // Also drain outstanding kernels during exception unwinding. Destructors
        // cannot throw; normal completion checks synchronization explicitly.
        cudaStreamSynchronize(stream);
        if (device) cudaFree(device);
        if (pinned) cudaFreeHost(pinned);
    }

    void allocate(int d)
    {
        check(cudaMalloc(&device, 5 * static_cast<size_t>(d) * sizeof(double)));
        check(cudaHostAlloc(&pinned, 256 * sizeof(double), cudaHostAllocMapped));
    }
};
} // namespace

// Run host-controlled PCG on a normalized RHS using the true Q(T) operator.
// Restore the original scale before returning lambda and convergence metadata.
PCGResult solve_backward_native(
    const double* T, const double* rhs, double* lambda, int n, int m,
    SPLRBackwardContext& context, BackwardPreconditioner mode,
    double tol, int max_iter, cudaStream_t stream
)
{
    // Validate dimensions, solver options and the retained context.
    validate_backward_dimensions(n, m);
    if (!std::isfinite(tol) || tol <= 0 || max_iter <= 0)
        throw std::invalid_argument("Native backward requires finite tol > 0 and max_iter > 0");
    const int d = n + (m - 1);
    if (!context.is_valid() || context.problem_size() != d)
        throw std::invalid_argument("Native backward context dimension/state mismatch");
    if (mode != BackwardPreconditioner::None && mode != BackwardPreconditioner::Diag &&
        mode != BackwardPreconditioner::Splr)
        throw std::invalid_argument("Invalid backward preconditioner");
    if (mode == BackwardPreconditioner::Diag && !context.has_diagonal_preconditioner())
        throw std::runtime_error("Backward diagonal preconditioner is not initialized");

    // Allocate the PCG vectors and final-plan marginal storage.
    Workspace workspace(stream);
    workspace.allocate(d);
    double* r = workspace.device;
    double* z = r + d;
    double* p = z + d;
    double* w = p + d;
    double* rows = w + d;
    double* cols = rows + n;

    // Start from the zero solution on the caller stream.
    const size_t bytes = static_cast<size_t>(d) * sizeof(double);
    check(cudaMemsetAsync(lambda, 0, bytes, stream));

    // Inspect RHS magnitude without squaring tiny or large values.
    const double rhs_max = backward_max_abs_cuda(rhs, workspace.pinned, d, stream);
    if (!std::isfinite(rhs_max))
        throw std::runtime_error("PCG RHS is non-finite");
    // Only an exactly zero RHS can return without a PCG update.
    if (rhs_max == 0.0) return {0, true};

    // Normalize before squared reductions; scalbn avoids an overflowing 1/s.
    const int exponent = std::ilogb(rhs_max);
    backward_normalize_rhs_cuda(rhs, r, exponent, d, stream);
    check(cudaGetLastError());

    // Restore the RHS scale on both convergence and iteration-limit returns.
    auto finish = [&](int iterations, bool converged) {
        backward_rescale_solution_cuda(lambda, exponent, tol, workspace.pinned, d, stream);
        return PCGResult{iterations, converged};
    };

    // The existing pinned helpers synchronize this exact stream. Check errors
    // too: upstream helpers do not themselves propagate CUDA return codes.
    auto norm = [&](const double* v) {
        const double value = compute_l2_norm_cuda(v, workspace.pinned, d, stream);
        check(cudaGetLastError());
        check(cudaStreamSynchronize(stream));
        if (!std::isfinite(value))
            throw std::runtime_error("PCG reduction is non-finite");
        if (value == 0.0 && backward_max_abs_cuda(v, workspace.pinned, d, stream) != 0.0)
            throw std::runtime_error("PCG residual norm underflow");
        return value;
    };

    auto dot = [&](const double* a, const double* b) {
        const double value = compute_dot_prod_cuda(a, b, workspace.pinned, d, stream);
        check(cudaGetLastError());
        check(cudaStreamSynchronize(stream));
        if (!std::isfinite(value))
            throw std::runtime_error("PCG reduction is non-finite");
        return value;
    };

    // A/B provide preconditioners only; Q(T) remains the solved operator.
    auto precondition = [&]() {
        switch (mode)
        {
        case BackwardPreconditioner::None:
            check(cudaMemcpyAsync(z, r, bytes, cudaMemcpyDeviceToDevice, stream));
            break;
        case BackwardPreconditioner::Diag:
            context.apply_diagonal_preconditioner(r, z, stream);
            break;
        case BackwardPreconditioner::Splr:
            context.apply_preconditioner(r, z, stream);
            break;
        }
        check(cudaGetLastError());
    };

    // Measure the normalized RHS for the relative stopping rule.
    const double bnorm = norm(r);

    // Precompute row and retained-column marginals of the final plan.
    backward_marginals_cuda(T, rows, cols, n, m, stream);
    check(cudaGetLastError());

    // Initialize z and the first preconditioned search direction p.
    precondition();
    check(cudaMemcpyAsync(p, z, bytes, cudaMemcpyDeviceToDevice, stream));
    double rz = dot(r, z);

    for (int k = 0; k < max_iter; ++k)
    {
        // Compute Q(T) * p with the final dense plan.
        backward_q_matvec_cuda(T, rows, cols, p, w, n, m, stream);
        check(cudaGetLastError());

        // Require a positive denominator for the PCG step.
        const double pw = dot(p, w);
        if (pw <= 0)
            throw std::runtime_error("PCG breakdown: p^T Q p <= 0 (matrix not SPD?)");

        // Update the solution and recursive residual with alpha = rz / pw.
        backward_update_xr_cuda(lambda, r, p, w, rz / pw, d, stream);
        check(cudaGetLastError());

        // Check the relative residual before building the next direction.
        if (norm(r) < tol * bnorm)
            return finish(k + 1, true);

        // Precondition the new residual and update p with beta = next_rz / rz.
        precondition();
        const double next_rz = dot(r, z);
        backward_update_p_cuda(p, z, next_rz / rz, d, stream);
        check(cudaGetLastError());
        rz = next_rz;
    }

    // Return the accumulated solution in the original RHS scale.
    return finish(max_iter, false);
}
