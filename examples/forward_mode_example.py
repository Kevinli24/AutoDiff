"""Evaluate a scalar expression and its two partial derivatives."""

from autodiff import Node, add, forward, log, mul, sin, sub


x1 = Node(2.0, name="x1")
x2 = Node(5.0, name="x2")
y = sub(add(log(x1), mul(x1, x2)), sin(x2))

forward(x1)
dy_dx1 = y.partial_derivative

forward(x2)
dy_dx2 = y.partial_derivative

print(f"f(2, 5) = {y.value:.12f}")
print(f"df/dx1  = {dy_dx1:.12f}")
print(f"df/dx2  = {dy_dx2:.12f}")
