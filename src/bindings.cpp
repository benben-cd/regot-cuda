#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// PyTorch interface declarations (only when PyTorch is available)
#if defined(TORCH_BUILD)

#include <torch/extension.h>
#include "backward_context.h"

py::dict torch_solve_backward_native(
    torch::Tensor T, torch::Tensor rhs, SPLRBackwardContext& context,
    const std::string& mode, double tol, int max_iter
);

py::dict torch_sinkhorn_bcd(
    torch::Tensor M, torch::Tensor a, torch::Tensor b,
    double reg, double tol, int max_iter, int verbose,
    const py::kwargs& kwargs
);

py::dict torch_sinkhorn_splr(
    torch::Tensor M, torch::Tensor a, torch::Tensor b,
    double reg, double tol, int max_iter, int verbose,
    const py::kwargs& kwargs
);

py::dict torch_sinkhorn_splr_save_ctx(
    torch::Tensor M, torch::Tensor a, torch::Tensor b,
    double reg, double tol, int max_iter, int verbose,
    const py::kwargs& kwargs
);

// Register the SPLRBackwardContext class (defined in solvers_torch.cpp)
void register_backward_context(py::module_& m);

#else

// Main solvers
py::dict sinkhorn_bcd(
    py::array_t<double> M,
    py::array_t<double> a,
    py::array_t<double> b,
    double reg,
    double tol,
    int max_iter,
    int verbose,
    const py::kwargs& kwargs
);

py::dict sinkhorn_splr(
    py::array_t<double> M,
    py::array_t<double> a,
    py::array_t<double> b,
    double reg,
    double tol,
    int max_iter,
    int verbose,
    const py::kwargs& kwargs
);

// Testing functions
py::dict test_T_computation_sparsify(
    py::array_t<double> alpha,
    py::array_t<double> beta,
    py::array_t<double> M,
    py::array_t<double> a,
    py::array_t<double> b,
    double reg,
    double shift,
    int K,
    int nrun
);

py::array_t<double> test_sparse_cholesky_solve(
    py::array_t<double> values,
    py::array_t<int> colind,
    py::array_t<int> rowptr,
    py::array_t<double> rhs
);

#endif

// PYBIND11 module definition
// MODULE_NAME is defined in setup.py
PYBIND11_MODULE(MODULE_NAME, m)
{
    m.doc() = "CUDA-accelerated Regularized Optimal Transport (RegOT-CUDA)";

#if defined(TORCH_BUILD)

    // PyTorch interface submodule (only when PyTorch is available)
    py::module m_torch = m.def_submodule("torch", "PyTorch interface.");
    m_torch.def("sinkhorn_bcd", &torch_sinkhorn_bcd,
        py::arg("M"), py::arg("a"), py::arg("b"), py::arg("reg"),
        py::arg("tol") = 1e-6, py::arg("max_iter") = 1000, py::arg("verbose") = 0,
        "Sinkhorn Block Coordinate Descent algorithm (CUDA implementation with PyTorch interface)");

    m_torch.def("sinkhorn_splr", &torch_sinkhorn_splr,
        py::arg("M"), py::arg("a"), py::arg("b"), py::arg("reg"),
        py::arg("tol") = 1e-6, py::arg("max_iter") = 1000, py::arg("verbose") = 0,
        "Sinkhorn SPLR algorithm (CUDA implementation with PyTorch interface)");

    // Backward context class and the backward-capable solver
    register_backward_context(m_torch);

    // Keep the native PCG boundary private to Python backward orchestration.
    m_torch.def("_solve_backward_native", &torch_solve_backward_native,
        py::arg("T"), py::arg("rhs"), py::arg("context"),
        py::arg("mode") = "diag", py::arg("tol") = 1e-6,
        py::arg("max_iter") = 2000,
        "Internal native reduced-Q PCG solver");

    // Export forward results together with the retained backward context.
    m_torch.def("sinkhorn_splr_save_ctx", &torch_sinkhorn_splr_save_ctx,
        py::arg("M"), py::arg("a"), py::arg("b"), py::arg("reg"),
        py::arg("tol") = 1e-6, py::arg("max_iter") = 1000, py::arg("verbose") = 0,
        "Sinkhorn SPLR algorithm with backward context export "
        "(CUDA implementation with PyTorch interface)");

#else

    // Numpy interface submodule
    py::module m_numpy = m.def_submodule("numpy", "Numpy interface.");
    m_numpy.def("sinkhorn_bcd", &sinkhorn_bcd,
        py::arg("M"), py::arg("a"), py::arg("b"), py::arg("reg"),
        py::arg("tol") = 1e-6, py::arg("max_iter") = 1000, py::arg("verbose") = 0,
        "Sinkhorn Block Coordinate Descent algorithm (CUDA implementation)");

    m_numpy.def("sinkhorn_splr", &sinkhorn_splr,
        py::arg("M"), py::arg("a"), py::arg("b"), py::arg("reg"),
        py::arg("tol") = 1e-6, py::arg("max_iter") = 1000, py::arg("verbose") = 0,
        "Sinkhorn SPLR algorithm (CUDA implementation)");

    // Submodule for test functions
    py::module m_tests = m.def_submodule("tests", "Test functions.");
    m_tests.def("test_T_computation_sparsify", &test_T_computation_sparsify,
        py::arg("alpha"), py::arg("beta"), py::arg("M"), py::arg("a"), py::arg("b"), py::arg("reg"), py::arg("shift"), py::arg("K"),
        py::arg("nrun") = 1,
        "Test T computation and sparsification (CUDA implementation)");

    m_tests.def("test_sparse_cholesky_solve", &test_sparse_cholesky_solve,
        py::arg("values"), py::arg("colind"), py::arg("rowptr"), py::arg("rhs"),
        "Test sparse Cholesky solver using cuDSS (CUDA implementation)");

#endif

}
