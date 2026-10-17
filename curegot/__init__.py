from ._cuda_loader import preload_cuda_runtime

preload_cuda_runtime()

# Load NumPy interface
from ._internal_numpy import *

# When the package is built using Pytorch, we typically
# need to import torch first to properly load some libraries
# such as libc10.so
try:
    import torch
    # Load PyTorch interface
    from ._internal_torch import *
except:
    pass

__version__ = "0.1.0"


# Differentiable SPLR OT Python API
try:
    from . import autograd, backward
    from .autograd import (
        splr_ot,
    )
    from .backward import solve_backward

    # `torch` here is the native pybind submodule exported by
    # curegot._internal_torch.
    torch.splr_ot = splr_ot
    torch.solve_backward = solve_backward

except ImportError:
    # Preserve the original optional-PyTorch behavior of curegot:
    # NumPy-only installations should still be importable.
    pass