"""Matrix-free reduced-gauge implicit differentiation (last beta fixed to zero)."""

import torch

__all__ = ["rhs_Q_scale", "assemble_grad"]


def rhs_Q_scale(G, T, n, m):
    """mu = [(G . T) 1_m, ((G . T)^T 1_n)_{-m}]  (stored scale, no 1/eta)."""
    GT = G * T
    return torch.cat([GT.sum(dim=1), GT.sum(dim=0)[:m - 1]])


def assemble_grad(T, G, lam, eta, n, m):
    """dL/dM = (1/eta) * T . (lam_alpha 1^T + 1 lam_beta^T - G)."""
    # Restore the full dual vector with beta_m fixed to zero.
    lam_alpha = lam[:n]
    lam_beta = torch.zeros(m, dtype=lam.dtype, device=lam.device)
    lam_beta[:m - 1] = lam[n:]

    # Assemble the cost gradient in the original regularization scale.
    return (T * (lam_alpha[:, None] + lam_beta[None, :] - G)) / eta
