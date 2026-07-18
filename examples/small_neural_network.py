"""Train a two-layer neural network on XOR."""

from autodiff import TENSOR_BACKEND, Tensor


features = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
targets = Tensor([[0.0], [1.0], [1.0], [0.0]])

w1 = Tensor([[0.8, -0.7, 0.4, 0.6], [-0.5, 0.9, 0.7, -0.8]], requires_grad=True)
b1 = Tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
w2 = Tensor([[0.6], [-0.8], [0.7], [-0.5]], requires_grad=True)
b2 = Tensor(0.0, requires_grad=True)
parameters = (w1, b1, w2, b2)

for epoch in range(1200):
    hidden = (features @ w1 + b1).tanh()
    probabilities = (hidden @ w2 + b2).sigmoid()
    loss = -(targets * probabilities.log() + (1.0 - targets) * (1.0 - probabilities).log()).mean()
    loss.backward()
    for parameter in parameters:
        parameter.step(0.5)
    if epoch % 300 == 0 or epoch == 1199:
        print(f"epoch={epoch:4d} loss={loss.item():.6f}")

print(f"backend={TENSOR_BACKEND}")
print(f"predictions={probabilities.data}")

