#include "autodiff/tensor.hpp"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <vector>

using autodiff::Tensor;

int main() {
    constexpr std::size_t size = 4096;
    constexpr int iterations = 500;
    std::vector<double> values(size);
    for (std::size_t index = 0; index < size; ++index) {
        values[index] = -1.0 + 2.0 * static_cast<double>(index) / static_cast<double>(size - 1);
    }

    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        Tensor x(values, {size}, true);
        auto expression = (x.sigmoid() * x + x.sin()).pow(2.0).mean();
        expression.backward();
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "iterations: " << iterations << '\n'
              << "elements/backward: " << size << '\n'
              << "elapsed: " << elapsed << "s\n"
              << "backward passes/s: " << iterations / elapsed << '\n';
}

