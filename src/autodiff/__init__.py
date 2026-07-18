"""Forward- and reverse-mode automatic differentiation."""

from .core import Node, add, differentiate, forward, log, mul, sin, sub, topological_order

try:
    from ._tensor import Tensor, tensor

    TENSOR_BACKEND = "cpp"
except ImportError:
    from .tensor import Tensor, tensor

    TENSOR_BACKEND = "python"

__all__ = [
    "Node",
    "Tensor",
    "TENSOR_BACKEND",
    "add",
    "differentiate",
    "forward",
    "log",
    "mul",
    "sin",
    "sub",
    "tensor",
    "topological_order",
]
