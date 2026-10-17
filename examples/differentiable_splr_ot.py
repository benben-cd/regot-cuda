"""Single-input implicit differentiation; requires CUDA float64."""
import torch
import curegot

generator = torch.Generator().manual_seed(42)
M = torch.rand(16, 19, generator=generator, dtype=torch.float64).cuda()
M.requires_grad_(True)
a = torch.full((16,), 1 / 16, dtype=M.dtype, device=M.device)
b = torch.full((19,), 1 / 19, dtype=M.dtype, device=M.device)
# Replace "diag" with "splr" to reuse the forward factorization and rank-2 state.
T = curegot.torch.splr_ot(M, a, b, eta=0.3,
                         backward_preconditioner="diag")
loss = T.square().sum()
loss.backward()
print("gradient norm:", M.grad.norm().item())
