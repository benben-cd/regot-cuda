"""Single-input implicit differentiation; requires CUDA float64."""
import torch
import curegot

generator = torch.Generator().manual_seed(42)
M = torch.rand(16, 19, generator=generator, dtype=torch.float64).cuda()
M.requires_grad_(True)
a = torch.full((16,), 1 / 16, dtype=M.dtype, device=M.device)
b = torch.full((19,), 1 / 19, dtype=M.dtype, device=M.device)
reg = 0.3

# Replace "diag" with "splr" to reuse the forward factorization and rank-2 state.
T = curegot.torch.sinkhorn_autograd(
    M, a, b, reg,
    tol=1e-6,
    max_iter=1000,
    verbose=0,
    backward_preconditioner="diag",
    backward_tol=1e-6,
    backward_max_iter=2000,
)
loss = T.square().sum()
loss.backward()
print("gradient norm:", M.grad.norm().item())
