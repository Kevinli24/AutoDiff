"""Train binary logistic regression with the reverse-mode tensor API."""

from autodiff import TENSOR_BACKEND, Tensor


features = Tensor(
    [[-2.0, -1.0], [-1.0, -2.0], [-1.0, -0.5], [1.0, 0.5], [1.0, 2.0], [2.0, 1.0]]
)
targets = Tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
weights = Tensor([[0.0], [0.0]], requires_grad=True)
bias = Tensor(0.0, requires_grad=True)

for epoch in range(200):
    probabilities = (features @ weights + bias).sigmoid()
    loss = -(targets * probabilities.log() + (1.0 - targets) * (1.0 - probabilities).log()).mean()
    loss.backward()
    weights.step(0.2)
    bias.step(0.2)
    if epoch % 50 == 0 or epoch == 199:
        print(f"epoch={epoch:3d} loss={loss.item():.6f}")

print(f"backend={TENSOR_BACKEND}")
print(f"weights={weights.data}")
print(f"bias={bias.data}")

