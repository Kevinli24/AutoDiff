import math
import unittest

from autodiff import Node, add, differentiate, forward, log, mul, sin, sub, topological_order


class ForwardModeExampleTests(unittest.TestCase):
    def setUp(self):
        self.x1 = Node(2.0, name="x1")
        self.x2 = Node(5.0, name="x2")
        self.y = sub(add(log(self.x1), mul(self.x1, self.x2)), sin(self.x2))

    def test_function_value(self):
        expected = math.log(2.0) + 2.0 * 5.0 - math.sin(5.0)
        self.assertAlmostEqual(self.y.value, expected)

    def test_partial_derivative_with_respect_to_x1(self):
        forward(self.x1)
        self.assertAlmostEqual(self.y.partial_derivative, 5.5)

    def test_partial_derivative_with_respect_to_x2(self):
        forward(self.x2)
        self.assertAlmostEqual(self.y.partial_derivative, 2.0 - math.cos(5.0))

    def test_repeated_passes_for_different_roots_are_independent(self):
        forward(self.x1)
        self.assertAlmostEqual(self.y.partial_derivative, 5.5)
        forward(self.x2)
        self.assertAlmostEqual(self.y.partial_derivative, 2.0 - math.cos(5.0))


class GraphTests(unittest.TestCase):
    def test_topological_order_places_parents_before_children(self):
        x = Node(3.0)
        y = Node(4.0)
        output = add(mul(x, y), sin(x))
        ordering = topological_order(x)
        positions = {node: i for i, node in enumerate(ordering)}

        self.assertIn(output, positions)
        for node in ordering:
            for child in node.child_nodes:
                if child in positions:
                    self.assertLess(positions[node], positions[child])

    def test_shared_parent_contributions_are_both_accumulated(self):
        x = Node(3.0)
        square = mul(x, x)
        self.assertAlmostEqual(differentiate(square, x), 6.0)

    def test_unreachable_output_has_zero_derivative(self):
        x = Node(2.0)
        y = Node(7.0)
        output = sin(y)
        self.assertEqual(differentiate(output, x), 0.0)

    def test_numeric_constants_and_operator_overloads(self):
        x = Node(4.0)
        output = 2.0 + x * 3.0 - 1.0
        self.assertEqual(output.value, 13.0)
        self.assertEqual(differentiate(output, x), 3.0)

    def test_cycle_is_reported(self):
        x = Node(1.0)
        y = sin(x)
        y.child_nodes.append(x)
        with self.assertRaisesRegex(ValueError, "cycle"):
            topological_order(x)


class LocalGradientTests(unittest.TestCase):
    def test_wrapping_functions_record_graph_metadata(self):
        x = Node(2.0)
        y = Node(5.0)

        product = mul(x, y)
        self.assertEqual(product.parent_nodes, [x, y])
        self.assertEqual(product.grad_wrt_parents, [5.0, 2.0])
        self.assertEqual(product.operator, "mul")
        self.assertIn(product, x.child_nodes)
        self.assertIn(product, y.child_nodes)

        self.assertEqual(add(x, y).grad_wrt_parents, [1.0, 1.0])
        self.assertEqual(sub(x, y).grad_wrt_parents, [1.0, -1.0])
        self.assertEqual(log(x).grad_wrt_parents, [0.5])
        self.assertAlmostEqual(sin(y).grad_wrt_parents[0], math.cos(5.0))


if __name__ == "__main__":
    unittest.main()
