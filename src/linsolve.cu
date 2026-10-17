#include "linsolve.h"

#include <cstdio>
#include <cstdlib>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// cuDSS error checking macro
#define CUDSS_CHECK(call) \
    do { \
        cudssStatus_t err = call; \
        if (err != CUDSS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDSS Error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


SparseCholeskySolver::SparseCholeskySolver()
{
    // Handle initialization
    CUDSS_CHECK(cudssCreate(&m_handle));

    // Configuration
    CUDSS_CHECK(cudssConfigCreate(&m_config));
    // Use CUDSS_ALG_DEFAULT
    cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
    CUDSS_CHECK(cudssConfigSet(
        m_config, CUDSS_CONFIG_REORDERING_ALG,
        (void*)&reorder_alg, sizeof(cudssAlgType_t)
    ));

    // Data object creation
    CUDSS_CHECK(cudssDataCreate(m_handle, &m_data));

    // Set CUDA stream
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    CUDSS_CHECK(cudssSetStream(m_handle, m_stream));
}

// Destroy all owned handles. Every call is guarded against nullptr so that
// a partially-initialized or moved-from solver is safe to destroy.
//
// Do not use CUDA_CHECK/CUDSS_CHECK here: destructors may run during CUDA
// context teardown, at which point CUDA/cuDSS calls may legitimately fail.
void SparseCholeskySolver::release()
{
    // Factorization and solve may be asynchronous. Make sure cuDSS has
    // stopped operating on the user buffers before any owner frees them.
    if (m_stream != nullptr)
    {
        cudaStreamSynchronize(m_stream);
    }

    // Matrix objects are lightweight wrappers around user buffers. Destroy
    // the wrappers before destroying the solver data and handle.
    if (m_A != nullptr)
    {
        cudssMatrixDestroy(m_A);
        m_A = nullptr;
    }
    if (m_x != nullptr)
    {
        cudssMatrixDestroy(m_x);
        m_x = nullptr;
    }
    if (m_b != nullptr)
    {
        cudssMatrixDestroy(m_b);
        m_b = nullptr;
    }

    if (m_data != nullptr && m_handle != nullptr)
    {
        cudssDataDestroy(m_handle, m_data);
        m_data = nullptr;
    }
    if (m_config != nullptr)
    {
        cudssConfigDestroy(m_config);
        m_config = nullptr;
    }
    if (m_handle != nullptr)
    {
        cudssDestroy(m_handle);
        m_handle = nullptr;
    }

    if (m_stream != nullptr)
    {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
}

SparseCholeskySolver::~SparseCholeskySolver()
{
    release();
}

SparseCholeskySolver::SparseCholeskySolver(SparseCholeskySolver&& other) noexcept
{
    // Steal all handles from the source object
    m_stream = other.m_stream;
    m_handle = other.m_handle;
    m_config = other.m_config;
    m_data   = other.m_data;
    m_A      = other.m_A;
    m_b      = other.m_b;
    m_x      = other.m_x;

    // Leave the source empty
    other.m_stream = nullptr;
    other.m_handle = nullptr;
    other.m_config = nullptr;
    other.m_data   = nullptr;
    other.m_A      = nullptr;
    other.m_b      = nullptr;
    other.m_x      = nullptr;
}

SparseCholeskySolver& SparseCholeskySolver::operator=(SparseCholeskySolver&& other) noexcept
{
    if (this != &other)
    {
        // Release the resources currently held. This also synchronizes the
        // current stream before destroying cuDSS objects.
        release();

        // Steal all handles from the source object
        m_stream = other.m_stream;
        m_handle = other.m_handle;
        m_config = other.m_config;
        m_data   = other.m_data;
        m_A      = other.m_A;
        m_b      = other.m_b;
        m_x      = other.m_x;

        // Leave the source empty
        other.m_stream = nullptr;
        other.m_handle = nullptr;
        other.m_config = nullptr;
        other.m_data   = nullptr;
        other.m_A      = nullptr;
        other.m_b      = nullptr;
        other.m_x      = nullptr;
    }
    return *this;
}

void SparseCholeskySolver::set_A(
    double* d_values, int* d_colind, int* d_rowptr,
    int n, size_t nnz
)
{
    // Replace semantics: each create must be paired with a destroy.
    if (m_A != nullptr)
    {
        CUDSS_CHECK(cudssMatrixDestroy(m_A));
        m_A = nullptr;
    }

    // CUDSS_MTYPE_SPD for Cholesky decomposition
    CUDSS_CHECK(cudssMatrixCreateCsr(
        &m_A, n, n, nnz,
        d_rowptr, nullptr, d_colind, d_values,
        CUDA_R_32I, CUDA_R_64F,
        CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO
    ));
}

void SparseCholeskySolver::set_b(double* d_rhs, int n, int nrhs)
{
    // Replace semantics: each create must be paired with a destroy.
    if (m_b != nullptr)
    {
        CUDSS_CHECK(cudssMatrixDestroy(m_b));
        m_b = nullptr;
    }

    CUDSS_CHECK(cudssMatrixCreateDn(
        &m_b, n, nrhs, n,
        d_rhs,
        CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
    ));
}

void SparseCholeskySolver::set_x(double* d_sol, int n, int nrhs)
{
    // Replace semantics: each create must be paired with a destroy.
    if (m_x != nullptr)
    {
        CUDSS_CHECK(cudssMatrixDestroy(m_x));
        m_x = nullptr;
    }

    CUDSS_CHECK(cudssMatrixCreateDn(
        &m_x, n, nrhs, n,
        d_sol,
        CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
    ));
}

void SparseCholeskySolver::reorder()
{
    CUDSS_CHECK(cudssExecute(
        m_handle, CUDSS_PHASE_REORDERING, m_config, m_data,
        m_A, m_x, m_b
    ));
}

void SparseCholeskySolver::symfac()
{
    CUDSS_CHECK(cudssExecute(
        m_handle, CUDSS_PHASE_SYMBOLIC_FACTORIZATION, m_config, m_data,
        m_A, m_x, m_b
    ));
}

void SparseCholeskySolver::analyze()
{
    CUDSS_CHECK(cudssExecute(
        m_handle, CUDSS_PHASE_ANALYSIS, m_config, m_data,
        m_A, m_x, m_b
    ));
}

void SparseCholeskySolver::factorize()
{
    CUDSS_CHECK(cudssExecute(
        m_handle, CUDSS_PHASE_FACTORIZATION, m_config, m_data,
        m_A, m_x, m_b
    ));
}

void SparseCholeskySolver::solve()
{
    CUDSS_CHECK(cudssExecute(
        m_handle, CUDSS_PHASE_SOLVE, m_config, m_data,
        m_A, m_x, m_b
    ));
}

// Sparse Cholesky solver using cuDSS
// Solves the linear system H * x = rhs where H is a symmetric positive definite matrix
// H is given in CSR format: values, colind, rowptr
// Only the lower triangular part is used
//
// In: d_values  [nnz]  -- non-zero values of H in CSR format
// In: d_colind  [nnz]  -- column indices
// In: d_rowptr  [n+1]  -- row pointers
// In: d_rhs     [n]    -- right-hand side vector
// Out: d_x      [n]    -- solution vector
// In: n         [int]  -- matrix dimension
// In: nnz       [int]  -- number of non-zero elements
void sparse_cholesky_solve(
    double* d_values,
    int* d_colind,
    int* d_rowptr,
    double* d_rhs,
    double* d_x,
    int n,
    int nnz
)
{
    // Create solver
    SparseCholeskySolver solver;

    // Matrix creation
    solver.set_A(d_values, d_colind, d_rowptr, n, nnz);
    solver.set_b(d_rhs, n, 1);
    solver.set_x(d_x, n, 1);

    // Step 1: Symbolic analysis
    solver.analyze();
    // Step 2: Factorization
    solver.factorize();
    // Step 3: Solve
    solver.solve();

    // The solver destructor synchronizes its stream before returning from
    // this scope, so d_x is ready for subsequent copies.
}

// Host function, mainly to test sparse Cholesky solver
void sparse_cholesky_solve_host(
    const double* values,
    const int* colind,
    const int* rowptr,
    const double* rhs,
    double* x,
    int n,
    int nnz
)
{
    // Allocate device memory
    double *d_values, *d_rhs, *d_x;
    int *d_colind, *d_rowptr;

    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_colind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rowptr, (n + 1) * sizeof(int)));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_values, values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs, rhs, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colind, colind, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowptr, rowptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Call device function
    sparse_cholesky_solve(d_values, d_colind, d_rowptr, d_rhs, d_x, n, nnz);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_colind));
    CUDA_CHECK(cudaFree(d_rowptr));
}
