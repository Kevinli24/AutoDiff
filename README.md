# AutoDiff

AutoDiff is my from-scratch implementation of the core machinery behind an
automatic-differentiation engine. I built it to understand how tensor operations
form a computational graph, how gradients flow through that graph, and how a
C++ numerical core can be exposed through a small Python API.

The project includes a scalar forward-mode implementation and a tensor-capable
reverse-mode engine. The native backend is written in C++17, with a NumPy
implementation serving as both a portable fallback and a readable reference.

A current limitation that I plan to work on in the future is having CPU-only
tensors.

## Features

- Scalar forward-mode differentiation with a topologically ordered graph
- Reverse-mode automatic differentiation and dynamic computational graphs
- Dense N-dimensional tensors with NumPy-style elementwise broadcasting
- Matrix multiplication, reductions, arithmetic, powers, and common activations
- A dependency-free C++17 API
- CPython bindings built directly with `setuptools`
- A NumPy reference/fallback backend when a C++ compiler is unavailable
- Numerical-gradient, graph, broadcasting, optimization, and C++ unit tests
- Python and C++ microbenchmarks
- Logistic-regression and two-layer neural-network training examples

## Python quick start

```python
from autodiff import Tensor

x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
w = Tensor([[2.0], [-1.0]], requires_grad=True)
loss = ((x @ w).sigmoid() ** 2).mean()
loss.backward()

print(loss.item())
print(x.grad)
print(w.grad)
```

Each result stores its parent tensors and creating operation. `backward()` walks
that graph in reverse topological order, seeds the output derivative with one,
and accumulates vector-Jacobian products into every reachable parameter.
Broadcast gradients are summed back to the original operand shapes.

Install in editable mode and build the compiled extension:

```console
python -m pip install -e .
```

If the extension is not present, importing `autodiff.Tensor` automatically uses
the NumPy reference backend. Inspect `autodiff.TENSOR_BACKEND` to see `"cpp"` or
`"python"`. Installation remains usable when no compiler is installed; set
`AUTODIFF_REQUIRE_CPP=1` if a missing or failed native build should be fatal.

## C++ API

```cpp
#include "autodiff/tensor.hpp"

using autodiff::Tensor;

Tensor x({1.0, 2.0, 3.0}, {3}, true);
auto loss = (x * x + x.sin()).mean();
loss.backward();

const auto& gradient = x.grad();
```

Build the static library, tests, and benchmark with CMake:

```console
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The C++ implementation lives in `include/autodiff/tensor.hpp` and
`src/cpp/tensor.cpp`. It uses standard-library containers and requires C++17.
The binding in `src/python_bindings.cpp` uses the CPython C API, so pybind11 is
not required.

## Design decisions

- Operations execute eagerly and record their parents, producing a dynamic graph
  that is rebuilt for every evaluation.
- Reverse-mode differentiation walks that graph in reverse topological order and
  accumulates each operation's local gradient into its parents.
- Broadcasted gradients are reduced back to the original operand shape before
  they are accumulated.
- The C++ backend uses only the standard library. I used the CPython C API for
  the binding to keep the native dependency surface small and to learn how Python
  extension types work below higher-level binding libraries.
- The NumPy backend mirrors the public tensor API so the package remains usable
  without a C++ compiler and provides a straightforward correctness reference.

## Current limitations

- Tensors are dense, CPU-only, and use double-precision floating-point values.
- Matrix multiplication currently accepts rank-2 tensors only.
- The project does not include GPU kernels, graph compilation, automatic
  batching, or higher-order gradients.
- `backward()` clears gradients on the reachable graph before each traversal;
  gradients do not accumulate across separate calls.
- This is an educational engine rather than a replacement for a production
  machine-learning framework.

## Operations

Both tensor backends provide:

- `+`, `-`, `*`, `/`, unary `-`, and scalar powers
- NumPy-style broadcasting for elementwise binary operations
- `@` / `matmul` for rank-2 matrix multiplication
- `sum()` and `mean()` reductions
- `log()`, `exp()`, `sin()`, `sigmoid()`, `relu()`, and `tanh()`
- `backward()`, `zero_grad()`, and `step(learning_rate)`

A non-scalar output requires an explicit seed gradient:

```python
y.backward([[1.0, 0.0], [0.0, 1.0]])
```

## Tests

Run the Python suite:

```console
python -m unittest discover -s tests -p "test_*.py" -v
```

`tests/test_tensor.py` compares a composite expression against central finite
differences and covers shared graph branches, broadcasting, matrix
multiplication, seeded backpropagation, and actual loss minimization.
`tests/test_tensor_cpp.cpp` exercises the native API and is registered with
CMake/CTest.

## Optimization examples

```console
python examples/logistic_regression.py
python examples/small_neural_network.py
```

Both examples rebuild the dynamic graph each training iteration, call
`loss.backward()`, and update leaf tensors with `step()`.

## Benchmarks

```console
python benchmarks/python_benchmark.py
cmake --build build --target autodiff_cpp_benchmark --config Release
```

The benchmarks report repeated construction and backward traversal of the same
elementwise workload. They are intentionally transparent microbenchmarks; use a
profiler and representative model for serious performance work.

## Scalar forward-mode API

Run the scalar differentiation example:

```console
python examples/forward_mode_example.py
```

The scalar API exports `Node`, `forward`, `differentiate`, `add`, `sub`, `mul`,
`log`, and `sin`.
