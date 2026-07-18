import math
import unittest

import numpy as np

from autodiff import TENSOR_BACKEND, Tensor


def array(value):
    return np.asarray(value, dtype=np.float64)


class ReverseModeTests(unittest.TestCase):
    def test_backend_is_identified(self):
        self.assertIn(TENSOR_BACKEND, {"cpp", "python"})

    def test_elementwise_chain_rule(self):
        x = Tensor([0.5, 1.5, -0.25], requires_grad=True)
        output = (x * x + x.sin()).sum()
        output.backward()

        expected = 2.0 * array([0.5, 1.5, -0.25]) + np.cos([0.5, 1.5, -0.25])
        np.testing.assert_allclose(array(x.grad), expected, rtol=1e-12, atol=1e-12)

    def test_shared_subexpression_accumulates_gradients(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        shared = x * x
        (shared + shared).sum().backward()
        np.testing.assert_allclose(array(x.grad), [8.0, 12.0])

    def test_broadcasting_reduces_gradient_to_operand_shape(self):
        matrix = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        bias = Tensor([0.1, 0.2, 0.3], requires_grad=True)
        (matrix + bias).sum().backward()
        np.testing.assert_allclose(array(matrix.grad), np.ones((2, 3)))
        np.testing.assert_allclose(array(bias.grad), [2.0, 2.0, 2.0])

    def test_matrix_multiplication_gradient(self):
        left = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        right = Tensor([[2.0], [-1.0]], requires_grad=True)
        (left @ right).sum().backward()
        np.testing.assert_allclose(array(left.grad), [[2.0, -1.0], [2.0, -1.0]])
        np.testing.assert_allclose(array(right.grad), [[4.0], [6.0]])

    def test_numerical_gradient_for_composite_expression(self):
        values = np.array([[0.2, -0.4], [1.1, 0.7]], dtype=np.float64)
        x = Tensor(values, requires_grad=True)
        loss = ((x.sigmoid() * x.exp()) / (x * x + 2.0)).mean()
        loss.backward()

        epsilon = 1e-6
        numerical = np.zeros_like(values)

        def evaluate(candidate):
            sigmoid = 1.0 / (1.0 + np.exp(-candidate))
            return np.mean(sigmoid * np.exp(candidate) / (candidate**2 + 2.0))

        for index in np.ndindex(values.shape):
            plus, minus = values.copy(), values.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numerical[index] = (evaluate(plus) - evaluate(minus)) / (2.0 * epsilon)

        np.testing.assert_allclose(array(x.grad), numerical, rtol=1e-5, atol=1e-7)

    def test_non_scalar_backward_accepts_explicit_seed(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        output = x * x
        output.backward([3.0, 4.0])
        np.testing.assert_allclose(array(x.grad), [6.0, 16.0])

    def test_non_scalar_backward_requires_seed(self):
        with self.assertRaises(ValueError):
            Tensor([1.0, 2.0], requires_grad=True).backward()

    def test_graph_metadata(self):
        x = Tensor([1.0], requires_grad=True)
        y = Tensor([2.0], requires_grad=True)
        output = x * y
        self.assertEqual(output.op, "mul")
        self.assertEqual(len(output.parents), 2)

    def test_gradient_descent_step(self):
        parameter = Tensor(3.0, requires_grad=True)
        (parameter**2).backward()
        parameter.step(0.1)
        self.assertAlmostEqual(float(array(parameter.data)), 2.4)


class OptimizationTests(unittest.TestCase):
    def test_logistic_regression_loss_decreases(self):
        features = Tensor([[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]])
        targets = Tensor([[0.0], [0.0], [1.0], [1.0]])
        weights = Tensor([[0.0], [0.0]], requires_grad=True)
        bias = Tensor(0.0, requires_grad=True)
        losses = []

        for _ in range(80):
            probabilities = (features @ weights + bias).sigmoid()
            loss = -(targets * probabilities.log() + (1.0 - targets) * (1.0 - probabilities).log()).mean()
            losses.append(loss.item())
            loss.backward()
            weights.step(0.2)
            bias.step(0.2)

        self.assertLess(losses[-1], losses[0] * 0.2)


if __name__ == "__main__":
    unittest.main()

