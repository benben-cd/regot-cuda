#pragma once

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>

#include "linsolve.h"

// Backward-propagation context for the SPLR solver.
//
// The backward pass of entropic-regularized OT requires solving linear
// systems with the reduced Hessian H = (1/reg) * Q(T*) by implicit
// differentiation. Depending on the ownership mode, the context either reuses
// the forward cuDSS factorization of the selected approximate Hessian A or
// factorizes a private CSR snapshot:
//
//     B = A + (reg/ys) * y * y^T - (A s)(A s)^T / (s^T A s)   [low_rank enabled]
//     B = A                                                    [low_rank disabled]
//
// where A = reg * H_Omega + (shift * reg) * I is the sparsified Hessian in
// the stored scale, s and y are the last secant pair, and ys = y^T s. The
// rank-2 solve uses the Woodbury formula
//
//     B^{-1} r = q - (s^T r / ys) * u
//                + [(1/reg + y^T u / ys) * (s^T r / ys) - y^T q / ys] * s
//
// with q = A^{-1} r and u = A^{-1} y. This is the forward search-direction
// update rule, minus the final -reg scaling.
//
// Ownership modes:
//
//   1. Immediate-convergence fallback: the forward provides a CSR
//      snapshot without low-rank materials; the context performs a fresh
//      analyze + factorization on its private buffers.
//   2. Adopted-forward-state mode (normal path): the context
//      takes ownership of the forward SparseCholeskySolver and the exact
//      backing buffers referenced by its cuDSS wrappers, then reuses the
//      existing forward factorization with zero extra factorization.
//
// Lifetime invariant:
//
//   1. m_solver is destroyed first, which destroys all cudssMatrix_t
//      wrappers, cudssData_t, cudssConfig_t and cudssHandle_t;
//   2. only after that are the CSR/rhs/solution/low-rank GPU buffers freed.
//
// In adopted mode, d_y and d_invA_y are non-owning aliases into the second
// columns of d_rhs and d_sol, respectively. Ownership flags below make this
// distinction explicit and prevent double-free in the destructor.
class SPLRBackwardContext
{
public:
    // problem_size = n + m - 1 (dimension of the reduced dual variable)
    // max_nnz      = Kmax + problem_size (capacity of the CSR buffers)
    SPLRBackwardContext(int problem_size, int max_nnz);
    ~SPLRBackwardContext();

    // Non-copyable: the context owns CUDA buffers and an opaque cuDSS state.
    SPLRBackwardContext(const SPLRBackwardContext&) = delete;
    SPLRBackwardContext& operator=(const SPLRBackwardContext&) = delete;

    // Movable: needed when returning the context through pybind. Ownership
    // of the solver and all buffers is transferred; the source becomes empty.
    SPLRBackwardContext(SPLRBackwardContext&& other) noexcept;
    SPLRBackwardContext& operator=(SPLRBackwardContext&& other) noexcept;

    // ---------------- Forward-side interface ----------------
    // The functions below are called by cuda_sinkhorn_splr_save_ctx() only.

    // Copy the final CSR for immediate-convergence fallback.
    // The normal path instead adopts the forward solver and its backing buffers.
    void snapshot_sphess(
        const double* d_values, const int* d_colind, const int* d_rowptr,
        size_t nnz
    );

    // Fallback path for immediate convergence: the snapshot was taken at the
    // final point and is factorized here without low-rank terms.
    void factorize_fallback(double reg);

    // Adopt a forward-owned solver and its exact backing buffers. This is the
    // normal-path ownership API; it reuses the existing forward
    // cuDSS factorization and performs no new analyze/factorization.
    //
    // Ownership transferred to this context:
    //   - solver (including its cuDSS factorization state);
    //   - d_values / d_colind / d_rowptr;
    //   - d_rhs_fwd / d_sol_fwd (each allocation has two columns);
    //   - d_s_fwd.
    //
    // In low-rank mode, d_y and d_invA_y are aliases:
    //   d_y      = d_rhs_fwd + problem_size
    //   d_invA_y = d_sol_fwd + problem_size
    // and are therefore not freed separately.
    void adopt_forward_state(
        std::unique_ptr<SparseCholeskySolver> solver,
        double* d_values, int* d_colind, int* d_rowptr,
        int dim, size_t nnz,
        double* d_rhs_fwd, double* d_sol_fwd,
        double* d_s_fwd,
        double ys, bool low_rank, double reg, bool fallback
    );

    // ---------------- Backward-side interface ----------------

    // Compute z = B^{-1} r. d_r and d_z are device pointers of length
    // problem_size. The cuDSS stream waits for caller_stream, then completes
    // the solve and copy-back before returning; scratch is safe to reuse.
    void apply_preconditioner(
        const double* d_r, double* d_z, cudaStream_t caller_stream
    );

    // ---------------- Native diagonal preconditioner ----------------
    // M_D^{-1} r = D_A^{-1} r, where D_A = diag(A) of the SAME stored-scale
    // sparsified matrix A whose factorization this context owns. The
    // exact-Q (unshifted) diagonal is intentionally NOT used here.
    //
    // Lifecycle: d_inv_diag is allocated once in the constructor (always
    // context-owned, like h_pinned -- never adopted, never aliased) and
    // filled once by setup_diagonal_preconditioner() after the CSR is
    // finalized (adopt / fallback all converge before this is
    // called). setup is idempotent; the PCG apply path below performs no
    // extraction, no allocation and no synchronization.

    // Extract D_A^{-1} from the owned CSR (d_Avalues/d_Acolind/d_Arowptr)
    // into d_inv_diag. One thread per row scans for col == row; every row
    // must have exactly one diagonal entry with a positive, finite value
    // (guaranteed by construction: sparsification keeps the diagonal and
    // the shift makes it positive; a violation here is a bug and fatal).
    // Synchronizes `stream` once to validate the result. One-time cost.
    void setup_diagonal_preconditioner(cudaStream_t stream);

    // z = D_A^{-1} r: single elementwise kernel on `stream`.
    // Fully asynchronous: no cudaMemcpy through d_rhs, no cuDSS, no stream
    // synchronization, no temporary allocation. Unlike
    // apply_preconditioner() (which synchronizes the solver stream
    // internally), this call returns after kernel launch; ordering with
    // the producer/consumer of d_r/d_z is inherited from `stream`, which
    // must therefore be the stream on which r was produced and z will be
    // consumed (the pybind wrapper passes the PyTorch current stream).
    void apply_diagonal_preconditioner(
        const double* d_r, double* d_z, cudaStream_t stream
    ) const;

    bool          has_diagonal_preconditioner() const { return m_inv_diag_ready; }
    // Metadata
    bool   is_valid()       const { return m_valid; }
    bool   is_fallback()    const { return m_fallback; }
    bool   low_rank()       const { return m_low_rank; }
    // true when the context adopted the forward solver's existing cuDSS
    // factorization; false for snapshot/refactorization fallback paths.
    bool   adopted_forward_state() const { return m_adopted_forward_state; }
    int    problem_size()   const { return m_Hsize; }
    double reg()            const { return m_reg; }
    double ys()             const { return m_ys; }

    // Validation-only accessors (device pointers owned by this context)
    const double* csr_values()  const { return d_Avalues; }
    const int*    csr_colind()  const { return d_Acolind; }
    const int*    csr_rowptr()  const { return d_Arowptr; }
    size_t        csr_nnz()     const { return m_nnz; }
    const double* secant_s()    const { return d_s; }
    const double* secant_y()    const { return d_y; }

private:
    // Free GPU buffers that can be replaced by adopted forward state. This
    // preserves d_inv_diag and h_pinned, which remain context-owned.
    void release_replaceable_buffers();

    // Free GPU/pinned buffers only. This function must not touch cuDSS
    // handles; the solver is reset separately before these buffers are freed.
    void release_buffers();

    // Create the solver if needed and factorize the owned CSR snapshot.
    void factorize_snapshot();

    int    m_Hsize = 0;        // n + m - 1
    int    m_max_nnz = 0;      // capacity of d_Avalues / d_Acolind
    size_t m_nnz = 0;          // actual nnz of the snapshot
    double m_reg = 0.0;        // regularization parameter (eta)
    double m_ys = 0.0;         // y^T s of the last secant pair
    bool   m_low_rank = false; // whether the rank-2 correction is active
    bool   m_fallback = false; // true if the fallback path was used
    bool   m_valid = false;    // true after the context has usable state
    bool   m_adopted_forward_state = false; // true after adopt_forward_state()

    // Allocation ownership flags. In snapshot mode all replaceable buffers
    // are context-owned. In adopted mode, CSR/rhs/sol/s are transferred and
    // owned, while y and invA_y are aliases into rhs/sol.
    bool m_owns_csr = false;
    bool m_owns_rhs = false;
    bool m_owns_sol = false;
    bool m_owns_s = false;
    bool m_owns_y = false;
    bool m_owns_invA_y = false;

    // Solver and active factorization state. A unique_ptr is used so that
    // destruction order is explicit: solver first, backing buffers second.
    std::unique_ptr<SparseCholeskySolver> m_solver;

    // CSR backing buffers (context-owned snapshot or adopted forward buffers)
    double* d_Avalues = nullptr;
    int*    d_Acolind = nullptr;
    int*    d_Arowptr = nullptr;

    // Fixed rhs/solution buffers, persistently bound to m_solver
    double* d_rhs = nullptr;
    double* d_sol = nullptr;

    // Low-rank materials. In adopted mode, d_y aliases d_rhs + m_Hsize and
    // d_invA_y aliases d_sol + m_Hsize; the ownership flags reflect this.
    double* d_s = nullptr;
    double* d_y = nullptr;
    double* d_invA_y = nullptr;


    // Diagonal preconditioner: D_A^{-1} of the owned CSR.
    // Always context-owned (allocated in the constructor, freed by
    // release_buffers, stolen by the move operations); never adopted, never
    // an alias of d_rhs/d_sol. Filled once by
    // setup_diagonal_preconditioner().
    double* d_inv_diag = nullptr;
    bool    m_inv_diag_ready = false;

    // Pinned host workspace for the reduction kernels
    double* h_pinned = nullptr;
};
