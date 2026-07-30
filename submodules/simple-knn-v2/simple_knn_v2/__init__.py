"""Python bindings for the simple-knn-v2 CUDA extension."""

# Load PyTorch and its shared libraries before importing the compiled extension.
import torch as _torch

from ._C import distCUDA2, distCUDACross

__all__ = ["distCUDA2", "distCUDACross"]
