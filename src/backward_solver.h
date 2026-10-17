#pragma once

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

// Legacy reductions use signed int grid-stride increments, up to 256 * 256.
constexpr int kMaxBackwardReducedDimension = INT_MAX - 256 * 256 + 1;

// Reject unsupported shapes before narrowing dimensions or allocating buffers.
inline void validate_backward_dimensions(std::int64_t n, std::int64_t m)
{
    // Bound each operand before adding, including inputs from int64 tensor shapes.
    if (n < 1 || m < 2 || n > kMaxBackwardReducedDimension ||
        m > kMaxBackwardReducedDimension || n + m - 1 > kMaxBackwardReducedDimension)
        throw std::invalid_argument(
            "Backward requires n >= 1, m >= 2 and n+m-1 <= kMaxBackwardReducedDimension (" +
            std::to_string(kMaxBackwardReducedDimension) + ")");
}

class SPLRBackwardContext;

enum class BackwardPreconditioner { None, Diag, Splr };

struct PCGResult
{
    int iterations;
    bool converged;
};

// Solve Q(T) lambda = rhs on the caller stream, with beta_m fixed to zero.
// The context's A/B are preconditioners only; T defines the true operator.
PCGResult solve_backward_native(
    const double* T, const double* rhs, double* lambda, int n, int m,
    SPLRBackwardContext& context, BackwardPreconditioner mode,
    double tol, int max_iter, cudaStream_t stream
);

// Compute full row marginals and the first m-1 column marginals of T.
void backward_marginals_cuda(
    const double* T, double* rows, double* cols, int n, int m, cudaStream_t stream
);

// Apply true reduced Q(T) using the cached marginals.
// v and out must be distinct allocations; T is row-major.
void backward_q_matvec_cuda(
    const double* T, const double* rows, const double* cols,
    const double* v, double* out, int n, int m, cudaStream_t stream
);

// Update the solution and recursive residual on the caller stream.
void backward_update_xr_cuda(
    double* x, double* r, const double* p, const double* w,
    double alpha, int d, cudaStream_t stream
);

// Build the next preconditioned search direction on the caller stream.
void backward_update_p_cuda(
    double* p, const double* z, double beta, int d, cudaStream_t stream
);

// Reduce max(abs(v)) into mapped pinned storage, then read on the host after
// checked caller-stream completion. Non-finite inputs produce infinity.
double backward_max_abs_cuda(
    const double* v, double* pinned, int d, cudaStream_t stream
);

// Scale rhs by 2^-exponent asynchronously on the caller stream.
void backward_normalize_rhs_cuda(
    const double* rhs, double* r, int exponent, int d, cudaStream_t stream
);

// Restore x's original scale; check output conversion after caller-stream
// completion using the solve's mapped pinned reduction buffer.
void backward_rescale_solution_cuda(
    double* x, int exponent, double tol, double* pinned, int d, cudaStream_t stream
);
