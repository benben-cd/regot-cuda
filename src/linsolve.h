#pragma once

#include <cstddef>
#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// cuDSS header
#include <cudss.h>

class SparseCholeskySolver
{
private:
    // CUDA stream
    cudaStream_t  m_stream = nullptr;
    // cuDSS data structures
    cudssHandle_t m_handle = nullptr;
    cudssConfig_t m_config = nullptr;
    cudssData_t   m_data = nullptr;
    cudssMatrix_t m_A = nullptr;
    cudssMatrix_t m_b = nullptr;
    cudssMatrix_t m_x = nullptr;

    // Destroy all owned cuDSS handles and the stream (null-safe).
    //
    // Resource order:
    //   1. synchronize the stream, because factorization/solve may be async;
    //   2. destroy matrix wrappers;
    //   3. destroy solver data/config/handle;
    //   4. destroy the stream.
    //
    // This class does not own the user buffers referenced by m_A/m_b/m_x.
    // The caller must keep those buffers alive until this solver has been
    // destroyed or explicitly released.
    void release();

public:
    SparseCholeskySolver();
    ~SparseCholeskySolver();

    // Non-copyable: copying opaque cuDSS handles would cause double-destroy.
    SparseCholeskySolver(const SparseCholeskySolver&) = delete;
    SparseCholeskySolver& operator=(const SparseCholeskySolver&) = delete;

    // Movable: transfers ownership of all cuDSS handles and the stream.
    // The moved-from object is left empty and is safe to destroy.
    SparseCholeskySolver(SparseCholeskySolver&& other) noexcept;
    SparseCholeskySolver& operator=(SparseCholeskySolver&& other) noexcept;

    // Get CUDA stream
    cudaStream_t get_cuda_stream() { return m_stream; }

    // Matrix creation/replacement.
    //
    // If a matrix object already exists, it is destroyed before the new one
    // is created. These functions may be called before a new ANALYSIS or
    // FACTORIZATION phase. They must not be interpreted as transferring an
    // existing factorization to a different matrix.
    void set_A(
        double* d_values, int* d_colind, int* d_rowptr,
        int n, size_t nnz
    );
    void set_b(double* d_rhs, int n, int nrhs = 1);
    void set_x(double* d_sol, int n, int nrhs = 1);

    // reorder() and symfac() are mainly for debugging purposes.
    // analyze() is basically a combination of reorder() and symfac().
    void reorder();
    void symfac();

    // Symbolic analysis
    void analyze();
    // Factorization
    void factorize();
    // Solve
    void solve();
};
