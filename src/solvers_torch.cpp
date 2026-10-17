// Only compile this file when PyTorch is available
#if defined(TORCH_BUILD)

#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "solvers.h"
#include "backward_context.h"
#include "backward_solver.h"
#include <cmath>

namespace py = pybind11;

// Validate Torch inputs and return native lambda / iteration / convergence data.
// All iterative algebra stays in the raw-pointer solver on the current stream.
py::dict torch_solve_backward_native(
    torch::Tensor T, torch::Tensor rhs, SPLRBackwardContext& context,
    const std::string& mode, double tol, int max_iter
)
{
    // Validate tensor shape, dtype, device and reduced-system dimensions.
    if (!T.is_cuda() || !rhs.is_cuda() || T.scalar_type() != torch::kFloat64 ||
        rhs.scalar_type() != torch::kFloat64 || T.dim() != 2 || rhs.dim() != 1)
        throw py::value_error("T/rhs must be CUDA float64 tensors of rank 2/1");
    const int64_t n = T.size(0), m = T.size(1);
    validate_backward_dimensions(n, m);
    if (T.device() != rhs.device() || rhs.numel() != n + m - 1)
        throw py::value_error("rhs must share T's device and have length n+m-1");
    if (!context.is_valid() || context.problem_size() != n + m - 1)
        throw py::value_error("Invalid or dimension-mismatched backward context");

    // Validate PCG options and select the native preconditioner.
    if (!std::isfinite(tol) || tol <= 0 || max_iter <= 0)
        throw py::value_error("tol must be finite and positive; max_iter must be positive");
    BackwardPreconditioner selected;
    if (mode == "none") selected = BackwardPreconditioner::None;
    else if (mode == "diag") selected = BackwardPreconditioner::Diag;
    else if (mode == "splr") selected = BackwardPreconditioner::Splr;
    else throw py::value_error("preconditioner must be none, diag or splr");

    // Guard T's device and verify that the retained context belongs to it.
    const at::cuda::CUDAGuard guard(T.device());
    cudaPointerAttributes attributes;
    const auto status = cudaPointerGetAttributes(&attributes, context.csr_values());
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    if (attributes.device != T.get_device())
        throw py::value_error("Backward context must be on T's device");

    // Capture the current PyTorch CUDA stream before tensor preparation.
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Keep contiguous inputs and output alive through the native solve.
    T = T.contiguous();
    rhs = rhs.contiguous();
    auto lambda = torch::empty_like(rhs);

    // Dispatch one native backward solve.
    const PCGResult result = solve_backward_native(
        T.data_ptr<double>(), rhs.data_ptr<double>(), lambda.data_ptr<double>(),
        static_cast<int>(n), static_cast<int>(m), context, selected, tol, max_iter, stream
    );

    // Expose the solution and convergence metadata to Python.
    py::dict out;
    out["lambda"] = lambda;
    out["iterations"] = result.iterations;
    out["converged"] = result.converged;
    return out;
}

// PyTorch interface for sinkhorn_bcd function
py::dict torch_sinkhorn_bcd(
    torch::Tensor M,
    torch::Tensor a,
    torch::Tensor b,
    double reg,
    double tol,
    int max_iter,
    int verbose,
    const py::kwargs& kwargs
)
{
    // Check input dimensions
    if (M.dim() != 2)
    {
        throw std::runtime_error("M must be a 2D tensor");
    }
    if (a.dim() != 1 || b.dim() != 1)
    {
        throw std::runtime_error("a and b must be 1D tensors");
    }

    int n = M.size(0);
    int m = M.size(1);

    if (a.size(0) != n || b.size(0) != m)
    {
        throw std::runtime_error("Shape mismatch: M is [n x m], a should be length n, b should be length m");
    }

    // Ensure that inputs are on GPU with a double type
    // Zero-copy if already satisfying the conditions
    auto src_options = M.options();
    auto src_device = M.device();
    // If M is already on GPU, preserve its device ID
    // Otherwise use the default device
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    if (src_device.is_cuda())
    {
        options = options.device(torch::kCUDA, src_device.index());
    }
    M = M.to(options);
    a = a.to(options);
    b = b.to(options);

    // Set active device
    const at::cuda::CUDAGuard device_guard(options.device());

    // Ensure contiguous and get data pointers
    M = M.contiguous();
    a = a.contiguous();
    b = b.contiguous();
    // Pointing to device memory
    const double* d_M_ptr = M.data_ptr<double>();
    const double* d_a_ptr = a.data_ptr<double>();
    const double* d_b_ptr = b.data_ptr<double>();

    // Handle kwargs for initial values (x0)
    const double* d_x0_ptr = nullptr;
    torch::Tensor x0_storage;

    if (kwargs.contains("x0"))
    {
        torch::Tensor x0 = py::cast<torch::Tensor>(kwargs["x0"]);
        if (x0.dim() != 1 || x0.size(0) != n + m)
        {
            throw std::runtime_error("x0 must be a 1D tensor of length n + m");
        }

        // Shallow copy x0 to x0_storage if the device, type, and layout of x0 are compatible
        // Otherwise make a conversion
        x0_storage = x0.to(options).contiguous();
        d_x0_ptr = x0_storage.data_ptr<double>();
    }

    // Create output tensors that are double and on device
    torch::Tensor P = torch::empty({n, m}, options);
    torch::Tensor dual = torch::empty(n + m, options);
    // Pointing to device memory
    double* d_P_ptr = P.data_ptr<double>();
    double* d_dual_ptr = dual.data_ptr<double>();

    // Call CUDA function for BCD algorithm
    int niter = 0;
    cuda_sinkhorn_bcd(
        d_M_ptr, d_a_ptr, d_b_ptr, d_P_ptr,
        reg, max_iter, tol, n, m, &niter,
        d_x0_ptr, d_dual_ptr,
        true, true
    );

    // Convert P and dual when necessary using the input tensor options
    P = P.to(src_options);
    dual = dual.to(src_options);

    // Create result dictionary
    py::dict result;
    result["niter"] = niter;
    result["plan"] = P;
    result["dual"] = dual;

    if (verbose > 0)
    {
        std::cout << "torch_sinkhorn_bcd (CUDA implementation) completed" << std::endl;
        std::cout << "Input shape: [" << n << " x " << m << "]" << std::endl;
        std::cout << "Regularization parameter: " << reg << std::endl;
        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Max iterations: " << max_iter << std::endl;
        std::cout << "Actual iterations: " << niter << std::endl;
        if (niter == max_iter)
        {
            std::cout << "Warning: Maximum iterations reached without convergence" << std::endl;
        }
    }

    return result;
}

// PyTorch interface for sinkhorn_splr function
py::dict torch_sinkhorn_splr(
    torch::Tensor M,
    torch::Tensor a,
    torch::Tensor b,
    double reg,
    double tol,
    int max_iter,
    int verbose,
    const py::kwargs& kwargs
)
{
    // Check input dimensions
    if (M.dim() != 2)
    {
        throw std::runtime_error("M must be a 2D tensor");
    }
    if (a.dim() != 1 || b.dim() != 1)
    {
        throw std::runtime_error("a and b must be 1D tensors");
    }

    int n = M.size(0);
    int m = M.size(1);

    if (a.size(0) != n || b.size(0) != m)
    {
        throw std::runtime_error("Shape mismatch: M is [n x m], a should be length n, b should be length m");
    }

    // Ensure that inputs are on GPU with a double type
    // Zero-copy if already satisfying the conditions
    auto src_options = M.options();
    auto src_device = M.device();
    // If M is already on GPU, preserve its device ID
    // Otherwise use the default device
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    if (src_device.is_cuda())
    {
        options = options.device(torch::kCUDA, src_device.index());
    }
    M = M.to(options);
    a = a.to(options);
    b = b.to(options);

    // Set active device
    const at::cuda::CUDAGuard device_guard(options.device());

    // Ensure contiguous and get data pointers
    M = M.contiguous();
    a = a.contiguous();
    b = b.contiguous();
    // Pointing to device memory
    const double* d_M_ptr = M.data_ptr<double>();
    const double* d_a_ptr = a.data_ptr<double>();
    const double* d_b_ptr = b.data_ptr<double>();

    // Handle kwargs for initial values (x0)
    const double* d_x0_ptr = nullptr;
    torch::Tensor x0_storage;

    if (kwargs.contains("x0"))
    {
        torch::Tensor x0 = py::cast<torch::Tensor>(kwargs["x0"]);
        if (x0.dim() != 1 || x0.size(0) != n + m)
        {
            throw std::runtime_error("x0 must be a 1D tensor of length n + m");
        }

        // Shallow copy x0 to x0_storage if the device, type, and layout of x0 are compatible
        // Otherwise make a conversion
        x0_storage = x0.to(options).contiguous();
        d_x0_ptr = x0_storage.data_ptr<double>();
    }

    // Get density_max from kwargs
    // Default to 10 / min(n, m)
    double density_max = 10.0 / std::min(n, m);
    density_max = std::min(density_max, 1.0);
    if (kwargs.contains("density"))
    {
        density_max = py::cast<double>(kwargs["density"]);
        density_max = std::min(density_max, 1.0);
        density_max = std::max(density_max, 0.0);
    }

    // Get shift_max from kwargs
    double shift_max = 0.001;
    if (kwargs.contains("shift"))
    {
        shift_max = py::cast<double>(kwargs["shift"]);
        shift_max = std::max(shift_max, 0.0);
    }

    // Get sparsity_pattern_cycle from kwargs
    int sparsity_pattern_cycle = 30;
    if (kwargs.contains("sparsity_pattern_cycle"))
    {
        sparsity_pattern_cycle = py::cast<int>(kwargs["sparsity_pattern_cycle"]);
    }

    // Get candidate_sinkhorn_iter from kwargs
    int candidate_sinkhorn_iter = 3;
    if (kwargs.contains("candidate_sinkhorn_iter"))
    {
        candidate_sinkhorn_iter = py::cast<int>(kwargs["candidate_sinkhorn_iter"]);
    }

    // Create output tensors that are double and on device
    torch::Tensor P = torch::empty({n, m}, options);
    torch::Tensor dual = torch::empty(n + m, options);
    // Pointing to device memory
    double* d_P_ptr = P.data_ptr<double>();
    double* d_dual_ptr = dual.data_ptr<double>();

    // Call CUDA function for SPLR algorithm
    int niter = 0;
    cuda_sinkhorn_splr(
        d_M_ptr, d_a_ptr, d_b_ptr, d_P_ptr,
        reg, max_iter, tol, n, m, &niter,
        density_max, shift_max,
        sparsity_pattern_cycle, candidate_sinkhorn_iter, verbose,
        d_x0_ptr, d_dual_ptr,
        true, true
    );

    // Convert P and dual when necessary using the input tensor options
    P = P.to(src_options);
    dual = dual.to(src_options);

    // Create result dictionary
    py::dict result;
    result["niter"] = niter;
    result["plan"] = P;
    result["dual"] = dual;

    if (verbose > 0)
    {
        std::cout << "torch_sinkhorn_splr (CUDA implementation) completed" << std::endl;
        std::cout << "Input shape: [" << n << " x " << m << "]" << std::endl;
        std::cout << "Regularization parameter: " << reg << std::endl;
        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Max iterations: " << max_iter << std::endl;
        std::cout << "Actual iterations: " << niter << std::endl;
        if (niter == max_iter)
        {
            std::cout << "Warning: Maximum iterations reached without convergence" << std::endl;
        }
    }

    return result;
}
// ---------------------------------------------------------------------------
// Backward context: pybind glue
// ---------------------------------------------------------------------------

// Apply the preconditioner: z = B^{-1} r, where B is the approximate Hessian
// (sparsified Hessian factorization plus the rank-2 BFGS correction) reused
// from the last forward iteration
torch::Tensor ctx_apply_preconditioner(SPLRBackwardContext& ctx, torch::Tensor r)
{
    if (!ctx.is_valid())
    {
        throw std::runtime_error("SPLRBackwardContext: invalid context");
    }
    if (r.dim() != 1 || r.size(0) != ctx.problem_size())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a 1D tensor of length n + m - 1");
    }
    if (!r.is_cuda())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a CUDA tensor");
    }

    // Ensure double type, preserving the device
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(r.device());
    r = r.to(options).contiguous();

    const at::cuda::CUDAGuard device_guard(r.device());
    torch::Tensor z = torch::empty(ctx.problem_size(), options);
    ctx.apply_preconditioner(
        r.data_ptr<double>(), z.data_ptr<double>(), at::cuda::getCurrentCUDAStream()
    );
    return z;
}

// Apply the diagonal preconditioner: z = D_A^{-1} r, where D_A = diag(A) of
// the context-owned CSR. Fully asynchronous on the PyTorch
// current stream -- unlike apply_preconditioner(), this performs no cuDSS
// call, no memcpy and no stream synchronization; ordering with the
// producer/consumer of r/z comes from the current stream itself.
torch::Tensor ctx_apply_diagonal_preconditioner(
    SPLRBackwardContext& ctx, torch::Tensor r
)
{
    if (!ctx.is_valid())
    {
        throw std::runtime_error("SPLRBackwardContext: invalid context");
    }
    if (!ctx.has_diagonal_preconditioner())
    {
        throw std::runtime_error(
            "SPLRBackwardContext: diagonal preconditioner not set up"
        );
    }
    if (r.dim() != 1 || r.size(0) != ctx.problem_size())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a 1D tensor of length n + m - 1");
    }
    if (!r.is_cuda())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a CUDA tensor");
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(r.device());
    r = r.to(options).contiguous();

    const at::cuda::CUDAGuard device_guard(r.device());
    torch::Tensor z = torch::empty(ctx.problem_size(), options);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ctx.apply_diagonal_preconditioner(
        r.data_ptr<double>(), z.data_ptr<double>(), stream
    );
    return z;
}

torch::Tensor ctx_csr_values(SPLRBackwardContext& ctx)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    torch::Tensor out = torch::empty({(long)ctx.csr_nnz()}, options);
    cudaMemcpy(out.data_ptr<double>(), ctx.csr_values(),
               ctx.csr_nnz() * sizeof(double), cudaMemcpyDeviceToDevice);
    return out;
}

torch::Tensor ctx_csr_colind(SPLRBackwardContext& ctx)
{
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({(long)ctx.csr_nnz()}, options);
    cudaMemcpy(out.data_ptr<int>(), ctx.csr_colind(),
               ctx.csr_nnz() * sizeof(int), cudaMemcpyDeviceToDevice);
    return out;
}

torch::Tensor ctx_csr_rowptr(SPLRBackwardContext& ctx)
{
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({(long)ctx.problem_size() + 1}, options);
    cudaMemcpy(out.data_ptr<int>(), ctx.csr_rowptr(),
               (ctx.problem_size() + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    return out;
}

torch::Tensor ctx_secant_s(SPLRBackwardContext& ctx)
{
    if (!ctx.low_rank())
    {
        throw std::runtime_error("SPLRBackwardContext: low-rank correction is disabled, s is unavailable");
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ctx.problem_size()}, options);
    cudaMemcpy(out.data_ptr<double>(), ctx.secant_s(),
               ctx.problem_size() * sizeof(double), cudaMemcpyDeviceToDevice);
    return out;
}

torch::Tensor ctx_secant_y(SPLRBackwardContext& ctx)
{
    if (!ctx.low_rank())
    {
        throw std::runtime_error("SPLRBackwardContext: low-rank correction is disabled, y is unavailable");
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ctx.problem_size()}, options);
    cudaMemcpy(out.data_ptr<double>(), ctx.secant_y(),
               ctx.problem_size() * sizeof(double), cudaMemcpyDeviceToDevice);
    return out;
}

// Register the SPLRBackwardContext class. Called from PYBIND11_MODULE.
// Metadata/accessors below are internal validation interfaces, not stable public API.
void register_backward_context(py::module_& m)
{
    py::class_<SPLRBackwardContext>(m, "SPLRBackwardContext")
        .def("apply_preconditioner", &ctx_apply_preconditioner, py::arg("r"))
        .def("apply_diagonal_preconditioner", &ctx_apply_diagonal_preconditioner, py::arg("r"))
        .def_property_readonly("has_diagonal_preconditioner", &SPLRBackwardContext::has_diagonal_preconditioner)
        .def_property_readonly("reg", &SPLRBackwardContext::reg)
        .def_property_readonly("dim", &SPLRBackwardContext::problem_size)
        // Internal validation/test metadata; not stable public API.
        .def_property_readonly("low_rank_enabled", &SPLRBackwardContext::low_rank)
        .def_property_readonly("is_fallback", &SPLRBackwardContext::is_fallback)
        .def_property_readonly("adopted_forward_state", &SPLRBackwardContext::adopted_forward_state)
        .def_property_readonly("ys", &SPLRBackwardContext::ys)
        .def("_csr_values", &ctx_csr_values)
        .def("_csr_colind", &ctx_csr_colind)
        .def("_csr_rowptr", &ctx_csr_rowptr)
        .def("_secant_s", &ctx_secant_s)
        .def("_secant_y", &ctx_secant_y);
}

py::dict torch_sinkhorn_splr_save_ctx(
    torch::Tensor M,
    torch::Tensor a,
    torch::Tensor b,
    double reg,
    double tol,
    int max_iter,
    int verbose,
    const py::kwargs& kwargs
)
{
    // Check input dimensions
    if (M.dim() != 2)
    {
        throw std::runtime_error("M must be a 2D tensor");
    }
    if (a.dim() != 1 || b.dim() != 1)
    {
        throw std::runtime_error("a and b must be 1D tensors");
    }

    // Check int64 shapes before narrowing or allocating an unsupported context.
    validate_backward_dimensions(M.size(0), M.size(1));
    const int n = static_cast<int>(M.size(0));
    const int m = static_cast<int>(M.size(1));

    if (a.size(0) != n || b.size(0) != m)
    {
        throw std::runtime_error("Shape mismatch: M is [n x m], a should be length n, b should be length m");
    }

    // Ensure that inputs are on GPU with a double type
    // Zero-copy if already satisfying the conditions
    auto src_options = M.options();
    auto src_device = M.device();
    // If M is already on GPU, preserve its device ID
    // Otherwise use the default device
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    if (src_device.is_cuda())
    {
        options = options.device(torch::kCUDA, src_device.index());
    }
    M = M.to(options);
    a = a.to(options);
    b = b.to(options);

    // Set active device
    const at::cuda::CUDAGuard device_guard(options.device());

    // Ensure contiguous and get data pointers
    M = M.contiguous();
    a = a.contiguous();
    b = b.contiguous();
    // Pointing to device memory
    const double* d_M_ptr = M.data_ptr<double>();
    const double* d_a_ptr = a.data_ptr<double>();
    const double* d_b_ptr = b.data_ptr<double>();

    // Handle kwargs for initial values (x0)
    const double* d_x0_ptr = nullptr;
    torch::Tensor x0_storage;

    if (kwargs.contains("x0"))
    {
        torch::Tensor x0 = py::cast<torch::Tensor>(kwargs["x0"]);
        if (x0.dim() != 1 || x0.size(0) != n + m)
        {
            throw std::runtime_error("x0 must be a 1D tensor of length n + m");
        }

        // Shallow copy x0 to x0_storage if the device, type, and layout of x0 are compatible
        // Otherwise make a conversion
        x0_storage = x0.to(options).contiguous();
        d_x0_ptr = x0_storage.data_ptr<double>();
    }

    // Get density_max from kwargs
    // Default to 10 / min(n, m)
    double density_max = 10.0 / std::min(n, m);
    density_max = std::min(density_max, 1.0);
    if (kwargs.contains("density"))
    {
        density_max = py::cast<double>(kwargs["density"]);
        density_max = std::min(density_max, 1.0);
        density_max = std::max(density_max, 0.0);
    }

    // Get shift_max from kwargs
    double shift_max = 0.001;
    if (kwargs.contains("shift"))
    {
        shift_max = py::cast<double>(kwargs["shift"]);
        shift_max = std::max(shift_max, 0.0);
    }

    // Get sparsity_pattern_cycle from kwargs
    int sparsity_pattern_cycle = 30;
    if (kwargs.contains("sparsity_pattern_cycle"))
    {
        sparsity_pattern_cycle = py::cast<int>(kwargs["sparsity_pattern_cycle"]);
    }

    // Get candidate_sinkhorn_iter from kwargs
    int candidate_sinkhorn_iter = 3;
    if (kwargs.contains("candidate_sinkhorn_iter"))
    {
        candidate_sinkhorn_iter = py::cast<int>(kwargs["candidate_sinkhorn_iter"]);
    }

    // Create output tensors that are double and on device
    torch::Tensor P = torch::empty({n, m}, options);
    torch::Tensor dual = torch::empty(n + m, options);
    // Pointing to device memory
    double* d_P_ptr = P.data_ptr<double>();
    double* d_dual_ptr = dual.data_ptr<double>();

    // Create the backward context with the same capacity as the forward
    // solver's CSR buffers (Kmax + Hsize)
    const int Hsize = n + m - 1;
    size_t Kmax = static_cast<size_t>(density_max * n * (m - 1));
    Kmax = std::max(Kmax, size_t(1));
    SPLRBackwardContext ctx(Hsize, (int)(Kmax + Hsize));

    // Call CUDA function for the backward-capable SPLR algorithm
    int niter = 0;
    bool converged = false;
    cuda_sinkhorn_splr_save_ctx(
        d_M_ptr, d_a_ptr, d_b_ptr, d_P_ptr,
        reg, max_iter, tol, n, m, &niter,
        density_max, shift_max,
        sparsity_pattern_cycle, candidate_sinkhorn_iter, verbose,
        d_x0_ptr, d_dual_ptr,
        true, true,
        &ctx, &converged
    );

    if (!ctx.is_valid())
    {
        throw std::runtime_error(
            "torch_sinkhorn_splr_save_ctx: forward pass failed (non-finite "
            "objective value); no backward context available"
        );
    }

    // one-time diagonal preconditioner setup. The context CSR is
    // final at this point on every path (adopt / fallback all
    // converge here), and all forward GPU work has host-completed (the
    // handoff paths end with synchronizing memcpys), so the extraction is
    // correctly ordered on the current stream. Fatal only on a structural
    // bug (missing / non-positive / non-finite diagonal).
    ctx.setup_diagonal_preconditioner(
        at::cuda::getCurrentCUDAStream()
    );

    // Convert P and dual when necessary using the input tensor options
    P = P.to(src_options);
    dual = dual.to(src_options);

    // Create result dictionary
    py::dict result;
    result["niter"] = niter;
    result["converged"] = converged;
    result["plan"] = P;
    result["dual"] = dual;
    result["context"] = py::cast(std::move(ctx));

    if (verbose > 0)
    {
        std::cout << "torch_sinkhorn_splr_save_ctx (CUDA implementation) completed" << std::endl;
        std::cout << "Input shape: [" << n << " x " << m << "]" << std::endl;
        std::cout << "Regularization parameter: " << reg << std::endl;
        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Max iterations: " << max_iter << std::endl;
        std::cout << "Actual iterations: " << niter << std::endl;
        if (niter == max_iter)
        {
            std::cout << "Warning: Maximum iterations reached without convergence" << std::endl;
        }
    }

    return result;
}


#endif  // #if defined(TORCH_BUILD)
