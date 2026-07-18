"""Scalar forward-mode automatic differentiation.

Each operation eagerly computes a scalar value and creates a graph node.  A
forward pass starts at one selected input, topologically orders its descendants,
and applies the chain rule using the local derivatives stored on every node.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, Optional, Union


Scalar = Union[int, float]
NodeLike = Union["Node", Scalar]


class Node:
    """One scalar expression in the computational graph.

    Root/input nodes normally need only a value; operation nodes are created by
    the wrapping functions in this module.
    """

    def __init__(
        self,
        value: Scalar,
        parent_nodes: Optional[Iterable["Node"]] = None,
        operator: Optional[str] = None,
        *,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(value, Real):
            raise TypeError(f"Node value must be a real number, got {type(value).__name__}")

        self.value = float(value)
        self.parent_nodes = list(parent_nodes or ())
        self.child_nodes: list[Node] = []
        self.operator = operator
        self.grad_wrt_parents: list[float] = []
        self.partial_derivative = 0.0
        self.name = name

    def __repr__(self) -> str:
        label = f", name={self.name!r}" if self.name is not None else ""
        operation = f", operator={self.operator!r}" if self.operator is not None else ""
        return f"Node(value={self.value!r}{operation}{label})"

    # Operator overloads keep graph construction readable in user code.
    def __add__(self, other: NodeLike) -> "Node":
        return add(self, other)

    def __radd__(self, other: NodeLike) -> "Node":
        return add(other, self)

    def __sub__(self, other: NodeLike) -> "Node":
        return sub(self, other)

    def __rsub__(self, other: NodeLike) -> "Node":
        return sub(other, self)

    def __mul__(self, other: NodeLike) -> "Node":
        return mul(self, other)

    def __rmul__(self, other: NodeLike) -> "Node":
        return mul(other, self)


def _as_node(value: NodeLike) -> Node:
    if isinstance(value, Node):
        return value
    if isinstance(value, Real):
        return Node(value)
    raise TypeError(f"operation operands must be Node or real numbers, got {type(value).__name__}")


def _operation(
    value: float,
    parents: list[Node],
    operator: str,
    gradients: list[float],
) -> Node:
    if len(parents) != len(gradients):
        raise ValueError("each parent must have one local derivative")

    result = Node(value, parents, operator)
    result.grad_wrt_parents = gradients
    for parent in parents:
        parent.child_nodes.append(result)
    return result


def add(node1: NodeLike, node2: NodeLike) -> Node:
    """Create the node ``node1 + node2``."""

    left, right = _as_node(node1), _as_node(node2)
    return _operation(left.value + right.value, [left, right], "add", [1.0, 1.0])


def sub(node1: NodeLike, node2: NodeLike) -> Node:
    """Create the node ``node1 - node2``."""

    left, right = _as_node(node1), _as_node(node2)
    return _operation(left.value - right.value, [left, right], "sub", [1.0, -1.0])


def mul(node1: NodeLike, node2: NodeLike) -> Node:
    """Create the node ``node1 * node2``."""

    left, right = _as_node(node1), _as_node(node2)
    return _operation(
        left.value * right.value,
        [left, right],
        "mul",
        [right.value, left.value],
    )


def log(node: NodeLike) -> Node:
    """Create the natural-logarithm node ``log(node)``."""

    parent = _as_node(node)
    return _operation(math.log(parent.value), [parent], "log", [1.0 / parent.value])


def sin(node: NodeLike) -> Node:
    """Create the node ``sin(node)`` (with angles measured in radians)."""

    parent = _as_node(node)
    return _operation(math.sin(parent.value), [parent], "sin", [math.cos(parent.value)])


def topological_order(root_node: Node) -> list[Node]:
    """Return a parent-before-child ordering of nodes reachable from ``root_node``.

    The traversal follows child links. An explicit active set provides a clear
    error if a caller manually corrupts the computational DAG by introducing a
    cycle.
    """

    if not isinstance(root_node, Node):
        raise TypeError("root_node must be a Node")

    ordering: list[Node] = []
    visited: set[Node] = set()
    active: set[Node] = set()

    def add_children(node: Node) -> None:
        if node in active:
            raise ValueError("the computational graph contains a cycle")
        if node in visited:
            return

        active.add(node)
        for child in node.child_nodes:
            add_children(child)
        active.remove(node)
        visited.add(node)
        ordering.append(node)

    add_children(root_node)
    ordering.reverse()
    return ordering


def forward(root_node: Node) -> list[Node]:
    """Compute derivatives with respect to ``root_node`` along its graph tape.

    Results are stored in each reachable node's ``partial_derivative`` member.
    The tape is returned for inspection.  Parents outside this tape contribute
    zero, which also makes repeated passes for different roots independent.
    """

    ordering = topological_order(root_node)
    reachable = set(ordering)

    for node in ordering:
        node.partial_derivative = 0.0
    root_node.partial_derivative = 1.0

    for node in ordering[1:]:
        if len(node.parent_nodes) != len(node.grad_wrt_parents):
            raise ValueError(
                f"node {node!r} has {len(node.parent_nodes)} parents but "
                f"{len(node.grad_wrt_parents)} local derivatives"
            )

        node.partial_derivative = sum(
            dnode_dparent * parent.partial_derivative
            for parent, dnode_dparent in zip(node.parent_nodes, node.grad_wrt_parents)
            if parent in reachable
        )

    return ordering


def differentiate(output_node: Node, root_node: Node) -> float:
    """Return ``d(output_node) / d(root_node)`` for the current graph values."""

    if not isinstance(output_node, Node):
        raise TypeError("output_node must be a Node")

    ordering = forward(root_node)
    if output_node not in set(ordering):
        return 0.0
    return output_node.partial_derivative
