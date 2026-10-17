#include "backward_solver.h"
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <string>

namespace {
constexpr int threads = 256;

struct Maximum
{
    template <typename T>
    __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};

// Compute row marginals or the alpha part of Q(T) * v.
// Full marginals include the final column; the cross-term omits fixed beta_m.
template <bool Marginals>
__global__ void rows_kernel(
    const double* T, const double* rows, const double* v, double* out, int n, int m
)
{
    using Reduce = cub::BlockReduce<double, threads>;
    __shared__ typename Reduce::TempStorage temp;

    // Blocks stride over rows; each row has one output writer.
    for (size_t i = blockIdx.x; i < static_cast<size_t>(n); i += gridDim.x)
    {
        // Accumulate the full marginal or retained-beta cross-term.
        double sum = 0.0;
        const int end = Marginals ? m : m - 1;
        for (size_t j = threadIdx.x; j < static_cast<size_t>(end); j += blockDim.x)
        {
            const double t = T[static_cast<size_t>(i) * m + j];
            sum += Marginals ? t : t * v[n + j];
        }

        // Reduce the row and add its diagonal contribution for Q(T) * v.
        const double total = Reduce(temp).Sum(sum);
        if (threadIdx.x == 0)
            out[i] = Marginals ? total : total + rows[i] * v[i];

        // Finish using CUB shared storage before processing the next row.
        __syncthreads();
    }
}

// Compute retained column marginals or the beta part of Q(T) * v.
// Each column has one writer, so no transpose or atomic reduction is needed.
template <bool Marginals>
__global__ void cols_kernel(
    const double* T, const double* cols, const double* v, double* out, int n, int m
)
{
    // Stride over the m-1 retained beta variables.
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x;
         j < static_cast<size_t>(m - 1); j += blockDim.x * gridDim.x)
    {
        // Accumulate this column's contribution from every alpha variable.
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double t = T[static_cast<size_t>(i) * m + j];
            sum += Marginals ? t : t * v[i];
        }

        // Write the marginal or add the retained-beta diagonal term.
        out[j] = Marginals ? sum : sum + cols[j] * v[n + j];
    }
}

// Update x and r with the current PCG step over the reduced vector.
__global__ void update_xr_kernel(
    double* x, double* r, const double* p, const double* w, double alpha, int d
)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < static_cast<size_t>(d); i += blockDim.x * gridDim.x)
    {
        x[i] += alpha * p[i];
        r[i] -= alpha * w[i];
    }
}

// Combine the preconditioned residual and previous direction in place.
__global__ void update_p_kernel(double* p, const double* z, double beta, int d)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < static_cast<size_t>(d); i += blockDim.x * gridDim.x)
        p[i] = z[i] + beta * p[i];
}

// Produce one maximum per block; map NaN/Inf to infinity for host rejection.
__global__ void max_abs_kernel(const double* v, double* partial, int d)
{
    using Reduce = cub::BlockReduce<double, threads>;
    __shared__ typename Reduce::TempStorage temp;

    // Traverse the reduced vector without squared reductions.
    double value = 0.0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < static_cast<size_t>(d); i += blockDim.x * gridDim.x)
        value = fmax(value, isfinite(v[i]) ? fabs(v[i]) : CUDART_INF);

    // Reduce and write disjoint block partials for the host.
    const double maximum = Reduce(temp).Reduce(value, Maximum());
    if (threadIdx.x == 0) partial[blockIdx.x] = maximum;
}

// Normalize the RHS directly by a power of two, without forming its reciprocal.
__global__ void normalize_rhs_kernel(const double* rhs, double* r, int exponent, int d)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < static_cast<size_t>(d); i += blockDim.x * gridDim.x)
        r[i] = scalbn(rhs[i], -exponent);
}

// Restore the solution scale and report conversion errors as block maxima.
__global__ void rescale_solution_kernel(
    double* x, int exponent, double tol, double* status, int d
)
{
    using Reduce = cub::BlockReduce<int, threads>;
    __shared__ typename Reduce::TempStorage temp;

    // Check each rescaled component while traversing the reduced vector.
    int error = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < static_cast<size_t>(d); i += blockDim.x * gridDim.x)
    {
        const double before = x[i];
        const double after = scalbn(before, exponent);
        if (!isfinite(before) || !isfinite(after)) error = 2;
        // Conservatively reject subnormal quantization/underflow that loses
        // more than tol per component when mapped back to the solved scale.
        else if (fabs(scalbn(after, -exponent) - before) > tol * fabs(before))
            error = max(error, 1);
        x[i] = after;
    }

    // Preserve error priority across each block before the host checks it.
    const int maximum = Reduce(temp).Reduce(error, Maximum());
    if (threadIdx.x == 0) status[blockIdx.x] = maximum;
}

void check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Native backward CUDA: ") + cudaGetErrorString(status));
}

int blocks(int count)
{
    return std::min(256, 1 + (count - 1) / threads);
}
} // namespace

// Cache Q(T)'s diagonal blocks on the supplied caller stream.
void backward_marginals_cuda(
    const double* T, double* rows, double* cols, int n, int m, cudaStream_t stream
)
{
    // Compute full row marginals, including the fixed-beta column.
    rows_kernel<true><<<std::min(n, 256), threads, 0, stream>>>(T, nullptr, nullptr, rows, n, m);

    // Compute only the retained column marginals.
    cols_kernel<true><<<blocks(m - 1), threads, 0, stream>>>(T, nullptr, nullptr, cols, n, m);
}

// Apply the true reduced operator; A/B preconditioners do not enter this step.
void backward_q_matvec_cuda(
    const double* T, const double* rows, const double* cols,
    const double* v, double* out, int n, int m, cudaStream_t stream
)
{
    // Write the alpha part on the supplied caller stream.
    rows_kernel<false><<<std::min(n, 256), threads, 0, stream>>>(T, rows, v, out, n, m);

    // Write the retained beta part to the disjoint output region.
    cols_kernel<false><<<blocks(m - 1), threads, 0, stream>>>(T, cols, v, out + n, n, m);
}

// Launch the coupled solution/residual update on the caller stream.
void backward_update_xr_cuda(
    double* x, double* r, const double* p, const double* w,
    double alpha, int d, cudaStream_t stream
)
{
    update_xr_kernel<<<blocks(d), threads, 0, stream>>>(x, r, p, w, alpha, d);
}

// Launch the next-direction update on the caller stream.
void backward_update_p_cuda(
    double* p, const double* z, double beta, int d, cudaStream_t stream
)
{
    update_p_kernel<<<blocks(d), threads, 0, stream>>>(p, z, beta, d);
}

// Return max(abs(v)) after checked caller-stream completion.
double backward_max_abs_cuda(const double* v, double* pinned, int d, cudaStream_t stream)
{
    // Reduce into the solve's mapped pinned buffer.
    double* mapped;
    check(cudaHostGetDevicePointer(&mapped, pinned, 0));
    const int count = blocks(d);
    max_abs_kernel<<<count, threads, 0, stream>>>(v, mapped, d);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));

    // Read partials only after the kernel completes.
    double maximum = 0.0;
    for (int i = 0; i < count; ++i) maximum = std::max(maximum, pinned[i]);
    return maximum;
}

// Enqueue normalization; the solver checks the launch and orders later work.
void backward_normalize_rhs_cuda(
    const double* rhs, double* r, int exponent, int d, cudaStream_t stream
)
{
    normalize_rhs_kernel<<<blocks(d), threads, 0, stream>>>(rhs, r, exponent, d);
}

// Rescale on the caller stream and reject unsafe conversion before returning.
void backward_rescale_solution_cuda(
    double* x, int exponent, double tol, double* pinned, int d, cudaStream_t stream
)
{
    // Restore the original magnitude and collect block error statuses.
    double* mapped;
    check(cudaHostGetDevicePointer(&mapped, pinned, 0));
    const int count = blocks(d);
    rescale_solution_kernel<<<count, threads, 0, stream>>>(x, exponent, tol, mapped, d);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));

    // Inspect statuses only after rescaling completes.
    int error = 0;
    for (int i = 0; i < count; ++i) error = std::max(error, static_cast<int>(pinned[i]));
    if (error == 2) throw std::runtime_error("PCG output rescaling overflow or non-finite solution");
    if (error == 1) throw std::runtime_error("PCG output rescaling loses precision (subnormal output)");
}
