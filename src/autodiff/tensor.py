"""NumPy reference backend for reverse-mode tensor automatic differentiation.

The public API mirrors the compiled C++ backend.  It is also useful as a
portable fallback when a C++17 compiler is not available during installation.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[float, int, Sequence[float], np.ndarray]
Axis = Optional[Union[int, Tuple[int, ...]]]


def _unbroadcast(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum a broadcast gradient back to an operand's original shape."""

    result = np.asarray(gradient, dtype=np.float64)
    while result.ndim > len(shape):
        result = result.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    return result.reshape(shape)


class Tensor:
    """A dense floating-point tensor and node in a reverse-mode graph."""

    __array_priority__ = 1000
    __hash__ = object.__hash__

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        *,
        _parents: Iterable["Tensor"] = (),
        _op: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self.parents = tuple(_parents)
        self.op = _op
        self._backward = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def __repr__(self) -> str:
        suffix = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data!r}{suffix})"

    @staticmethod
    def _coerce(value: Union["Tensor", ArrayLike]) -> "Tensor":
        return value if isinstance(value, Tensor) else Tensor(value)

    @staticmethod
    def _result(data: np.ndarray, parents: tuple["Tensor", ...], op: str) -> "Tensor":
        return Tensor(data, any(parent.requires_grad for parent in parents), _parents=parents, _op=op)

    def __add__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other = self._coerce(other)
        output = self._result(self.data + other.data, (self, other), "add")

        def backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(output.grad, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(output.grad, other.shape)

        output._backward = backward
        return output

    __radd__ = __add__

    def __neg__(self) -> "Tensor":
        output = self._result(-self.data, (self,), "neg")

        def backward() -> None:
            if self.requires_grad:
                self.grad -= output.grad

        output._backward = backward
        return output

    def __sub__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self + -self._coerce(other)

    def __rsub__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self._coerce(other) + -self

    def __mul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other = self._coerce(other)
        output = self._result(self.data * other.data, (self, other), "mul")

        def backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(output.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(output.grad * self.data, other.shape)

        output._backward = backward
        return output

    __rmul__ = __mul__

    def __truediv__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other = self._coerce(other)
        output = self._result(self.data / other.data, (self, other), "div")

        def backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(output.grad / other.data, self.shape)
            if other.requires_grad:
                other.grad -= _unbroadcast(output.grad * self.data / (other.data**2), other.shape)

        output._backward = backward
        return output

    def __rtruediv__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self._coerce(other) / self

    def __pow__(self, exponent: float) -> "Tensor":
        if not np.isscalar(exponent):
            return NotImplemented
        value = float(exponent)
        output = self._result(self.data**value, (self,), "pow")

        def backward() -> None:
            if self.requires_grad:
                self.grad += output.grad * value * self.data ** (value - 1.0)

        output._backward = backward
        return output

    def __matmul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other = self._coerce(other)
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul currently requires two rank-2 tensors")
        output = self._result(self.data @ other.data, (self, other), "matmul")

        def backward() -> None:
            if self.requires_grad:
                self.grad += output.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ output.grad

        output._backward = backward
        return output

    def sum(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        output = self._result(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def backward() -> None:
            if not self.requires_grad:
                return
            gradient = output.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                axes = tuple(item if item >= 0 else item + self.ndim for item in axes)
                for item in sorted(axes):
                    gradient = np.expand_dims(gradient, item)
            self.grad += np.broadcast_to(gradient, self.shape)

        output._backward = backward
        return output

    def mean(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            divisor = self.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            divisor = int(np.prod([self.shape[item] for item in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / divisor

    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        output = self._result(self.data.reshape(shape), (self,), "reshape")

        def backward() -> None:
            if self.requires_grad:
                self.grad += output.grad.reshape(self.shape)

        output._backward = backward
        return output

    @property
    def T(self) -> "Tensor":
        output = self._result(self.data.T, (self,), "transpose")

        def backward() -> None:
            if self.requires_grad:
                self.grad += output.grad.T

        output._backward = backward
        return output

    def log(self) -> "Tensor":
        output = self._result(np.log(self.data), (self,), "log")

        def backward() -> None:
            if self.requires_grad:
                self.grad += output.grad / self.data

        output._backward = backward
        return output

    def exp(self) -> "Tensor":
        output = self._result(np.exp(self.data), (self,), "exp")

        def backward() -> None:
            if self.requires_grad:
                self.grad += output.grad * output.data

        output._backward = backward
        return output

    def sin(self) -> "Tensor":
        output = self._result(np.sin(self.data), (self,), "sin")

        def backward() -> None:
            if self.requires_grad:
                self.grad += output.grad * np.cos(self.data)

        output._backward = backward
        return output

    def sigmoid(self) -> "Tensor":
        positive = self.data >= 0
        values = np.empty_like(self.data)
        values[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        exp_values = np.exp(self.data[~positive])
        values[~positive] = exp_values / (1.0 + exp_values)
        output = self._result(values, (self,), "sigmoid")

        def backward() -> None:
            if self.requires_grad:
                self.grad += output.grad * output.data * (1.0 - output.data)

        output._backward = backward
        return output

    def relu(self) -> "Tensor":
        output = self._result(np.maximum(self.data, 0.0), (self,), "relu")

        def backward() -> None:
            if self.requires_grad:
                self.grad += output.grad * (self.data > 0.0)

        output._backward = backward
        return output

    def tanh(self) -> "Tensor":
        output = self._result(np.tanh(self.data), (self,), "tanh")

        def backward() -> None:
            if self.requires_grad:
                self.grad += output.grad * (1.0 - output.data**2)

        output._backward = backward
        return output

    def backward(self, gradient: Optional[ArrayLike] = None) -> None:
        """Backpropagate from this tensor through its recorded graph."""

        if gradient is None:
            if self.size != 1:
                raise ValueError("a gradient must be supplied for non-scalar outputs")
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(gradient, dtype=np.float64)
            if seed.shape != self.shape:
                raise ValueError(f"gradient shape {seed.shape} does not match output shape {self.shape}")

        ordering: list[Tensor] = []
        visited: set[Tensor] = set()
        active: set[Tensor] = set()

        def visit(node: Tensor) -> None:
            if node in active:
                raise ValueError("the computational graph contains a cycle")
            if node in visited:
                return
            active.add(node)
            for parent in node.parents:
                visit(parent)
            active.remove(node)
            visited.add(node)
            ordering.append(node)

        visit(self)
        for node in ordering:
            if node.requires_grad:
                node.grad.fill(0.0)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad[...] = seed
        for node in reversed(ordering):
            node._backward()

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    def step(self, learning_rate: float) -> None:
        """Apply one in-place gradient-descent update to this tensor."""

        if not self.requires_grad or self.grad is None:
            raise ValueError("step requires a tensor with requires_grad=True")
        self.data -= float(learning_rate) * self.grad

    def item(self) -> float:
        if self.size != 1:
            raise ValueError("item() requires a tensor containing one value")
        return float(self.data.item())

    def numpy(self) -> np.ndarray:
        return self.data.copy()


def tensor(data: ArrayLike, requires_grad: bool = False) -> Tensor:
    """Construct a :class:`Tensor`."""

    return Tensor(data, requires_grad=requires_grad)

