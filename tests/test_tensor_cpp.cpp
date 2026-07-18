#include "autodiff/tensor.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using autodiff::Tensor;

bool close(double left, double right, double tolerance = 1e-10) {
    return std::abs(left - right) < tolerance;
}

int main() {
    Tensor x({2.0, 3.0}, {2}, true);
    auto square_sum = (x * x).sum();
    square_sum.backward();
    assert(close(square_sum.item(), 13.0));
    assert(close(x.grad()[0], 4.0));
    assert(close(x.grad()[1], 6.0));

    Tensor matrix({1, 2, 3, 4, 5, 6}, {2, 3}, true);
    Tensor bias({0.1, 0.2, 0.3}, {3}, true);
    (matrix + bias).sum().backward();
    for (double gradient : matrix.grad()) assert(close(gradient, 1.0));
    for (double gradient : bias.grad()) assert(close(gradient, 2.0));

    Tensor left({1, 2, 3, 4}, {2, 2}, true);
    Tensor right({2, -1}, {2, 1}, true);
    left.matmul(right).sum().backward();
    const std::vector<double> expected_left{2, -1, 2, -1};
    const std::vector<double> expected_right{4, 6};
    for (std::size_t i = 0; i < expected_left.size(); ++i) assert(close(left.grad()[i], expected_left[i]));
    for (std::size_t i = 0; i < expected_right.size(); ++i) assert(close(right.grad()[i], expected_right[i]));

    std::cout << "C++ tensor tests passed\n";
}

