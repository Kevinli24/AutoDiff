"""Small repeatable backward-pass benchmark; not a substitute for a profiler."""

from time import perf_counter

import numpy as np

from autodiff import TENSOR_BACKEND, Tensor


SIZE = 4096
ITERATIONS = 200
values = np.linspace(-1.0, 1.0, SIZE)

start = perf_counter()
for _ in range(ITERATIONS):
    x = Tensor(values, requires_grad=True)
    ((x.sigmoid() * x + x.sin()) ** 2).mean().backward()
elapsed = perf_counter() - start

print(f"backend: {TENSOR_BACKEND}")
print(f"iterations: {ITERATIONS}")
print(f"elements/backward: {SIZE}")
print(f"elapsed: {elapsed:.3f}s")
print(f"backward passes/s: {ITERATIONS / elapsed:.1f}")

