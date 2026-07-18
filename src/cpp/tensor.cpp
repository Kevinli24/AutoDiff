#include "autodiff/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace autodiff {
namespace detail {

struct Node {
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<std::size_t> shape;
    bool requires_grad = false;
    std::string op;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void(const std::vector<double>&)> backward;
};

}  // namespace detail

namespace {

using NodePtr = std::shared_ptr<detail::Node>;

std::size_t element_count(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                           std::multiplies<std::size_t>());
}

std::vector<std::size_t> strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> result(shape.size(), 1);
    for (std::size_t index = shape.size(); index-- > 1;) {
        result[index - 1] = result[index] * shape[index];
    }
    return result;
}

std::vector<std::size_t> broadcast_shape(const std::vector<std::size_t>& left,
                                         const std::vector<std::size_t>& right) {
    const auto rank = std::max(left.size(), right.size());
    std::vector<std::size_t> result(rank, 1);
    for (std::size_t offset = 0; offset < rank; ++offset) {
        const auto left_size = offset < left.size() ? left[left.size() - 1 - offset] : 1;
        const auto right_size = offset < right.size() ? right[right.size() - 1 - offset] : 1;
        if (left_size != right_size && left_size != 1 && right_size != 1) {
            throw std::invalid_argument("tensor shapes are not broadcast-compatible");
        }
        result[rank - 1 - offset] = std::max(left_size, right_size);
    }
    return result;
}

std::size_t broadcast_index(std::size_t output_index,
                            const std::vector<std::size_t>& output_shape,
                            const std::vector<std::size_t>& input_shape) {
    if (input_shape.empty()) {
        return 0;
    }
    const auto output_strides = strides(output_shape);
    const auto input_strides = strides(input_shape);
    const auto leading = output_shape.size() - input_shape.size();
    std::size_t input_index = 0;
    for (std::size_t axis = 0; axis < output_shape.size(); ++axis) {
        const auto coordinate = (output_index / output_strides[axis]) % output_shape[axis];
        if (axis >= leading) {
            const auto input_axis = axis - leading;
            if (input_shape[input_axis] != 1) {
                input_index += coordinate * input_strides[input_axis];
            }
        }
    }
    return input_index;
}

NodePtr make_node(std::vector<double> data, std::vector<std::size_t> shape,
                  bool requires_grad, std::string op = {},
                  std::vector<NodePtr> parents = {}) {
    auto node = std::make_shared<detail::Node>();
    node->data = std::move(data);
    node->grad.assign(node->data.size(), 0.0);
    node->shape = std::move(shape);
    node->requires_grad = requires_grad;
    node->op = std::move(op);
    node->parents = std::move(parents);
    return node;
}

enum class BinaryOp { Add, Subtract, Multiply, Divide };

Tensor binary(const NodePtr& left, const NodePtr& right, BinaryOp operation,
              const std::function<Tensor(NodePtr)>& wrap) {
    auto shape = broadcast_shape(left->shape, right->shape);
    const auto count = element_count(shape);
    std::vector<double> values(count);
    std::vector<std::size_t> left_indices(count), right_indices(count);

    for (std::size_t index = 0; index < count; ++index) {
        const auto li = broadcast_index(index, shape, left->shape);
        const auto ri = broadcast_index(index, shape, right->shape);
        left_indices[index] = li;
        right_indices[index] = ri;
        switch (operation) {
            case BinaryOp::Add: values[index] = left->data[li] + right->data[ri]; break;
            case BinaryOp::Subtract: values[index] = left->data[li] - right->data[ri]; break;
            case BinaryOp::Multiply: values[index] = left->data[li] * right->data[ri]; break;
            case BinaryOp::Divide: values[index] = left->data[li] / right->data[ri]; break;
        }
    }

    const char* name = operation == BinaryOp::Add ? "add" :
                       operation == BinaryOp::Subtract ? "sub" :
                       operation == BinaryOp::Multiply ? "mul" : "div";
    auto output = make_node(std::move(values), std::move(shape),
                            left->requires_grad || right->requires_grad,
                            name, {left, right});
    output->backward = [left, right, left_indices = std::move(left_indices),
                        right_indices = std::move(right_indices), operation]
                       (const std::vector<double>& upstream) {
        for (std::size_t index = 0; index < upstream.size(); ++index) {
            const auto li = left_indices[index];
            const auto ri = right_indices[index];
            if (left->requires_grad) {
                double local = 1.0;
                if (operation == BinaryOp::Multiply) local = right->data[ri];
                if (operation == BinaryOp::Divide) local = 1.0 / right->data[ri];
                left->grad[li] += upstream[index] * local;
            }
            if (right->requires_grad) {
                double local = operation == BinaryOp::Subtract ? -1.0 : 1.0;
                if (operation == BinaryOp::Multiply) local = left->data[li];
                if (operation == BinaryOp::Divide) {
                    local = -left->data[li] / (right->data[ri] * right->data[ri]);
                }
                right->grad[ri] += upstream[index] * local;
            }
        }
    };
    return wrap(std::move(output));
}

}  // namespace

struct TensorAccess {
    static Tensor wrap(NodePtr node) { return Tensor(std::move(node)); }
    static const NodePtr& node(const Tensor& tensor) { return tensor.node_; }
};

Tensor::Tensor() : Tensor(0.0, false) {}

Tensor::Tensor(double value, bool requires_grad)
    : node_(make_node({value}, {}, requires_grad)) {}

Tensor::Tensor(std::vector<double> data, std::vector<std::size_t> shape,
               bool requires_grad) {
    if (element_count(shape) != data.size()) {
        throw std::invalid_argument("data length does not match tensor shape");
    }
    for (auto dimension : shape) {
        if (dimension == 0) throw std::invalid_argument("tensor dimensions must be positive");
    }
    node_ = make_node(std::move(data), std::move(shape), requires_grad);
}

Tensor::Tensor(NodePtr node) : node_(std::move(node)) {}

const std::vector<double>& Tensor::data() const { return node_->data; }
const std::vector<double>& Tensor::grad() const { return node_->grad; }
const std::vector<std::size_t>& Tensor::shape() const { return node_->shape; }
std::size_t Tensor::size() const { return node_->data.size(); }
bool Tensor::requires_grad() const { return node_->requires_grad; }
const std::string& Tensor::op() const { return node_->op; }

std::vector<Tensor> Tensor::parents() const {
    std::vector<Tensor> result;
    result.reserve(node_->parents.size());
    for (const auto& parent : node_->parents) result.emplace_back(TensorAccess::wrap(parent));
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    return binary(node_, other.node_, BinaryOp::Add, TensorAccess::wrap);
}
Tensor Tensor::operator-(const Tensor& other) const {
    return binary(node_, other.node_, BinaryOp::Subtract, TensorAccess::wrap);
}
Tensor Tensor::operator*(const Tensor& other) const {
    return binary(node_, other.node_, BinaryOp::Multiply, TensorAccess::wrap);
}
Tensor Tensor::operator/(const Tensor& other) const {
    return binary(node_, other.node_, BinaryOp::Divide, TensorAccess::wrap);
}
Tensor Tensor::operator-() const { return Tensor(0.0) - *this; }

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape().size() != 2 || other.shape().size() != 2) {
        throw std::invalid_argument("matmul requires two rank-2 tensors");
    }
    const auto rows = shape()[0], inner = shape()[1], columns = other.shape()[1];
    if (inner != other.shape()[0]) throw std::invalid_argument("matmul inner dimensions differ");
    std::vector<double> values(rows * columns, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            for (std::size_t k = 0; k < inner; ++k) {
                values[row * columns + column] += data()[row * inner + k] * other.data()[k * columns + column];
            }
        }
    }
    auto left = node_, right = other.node_;
    auto output = make_node(std::move(values), {rows, columns},
                            requires_grad() || other.requires_grad(), "matmul", {left, right});
    output->backward = [left, right, rows, inner, columns](const std::vector<double>& upstream) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const auto gradient = upstream[row * columns + column];
                for (std::size_t k = 0; k < inner; ++k) {
                    if (left->requires_grad) left->grad[row * inner + k] += gradient * right->data[k * columns + column];
                    if (right->requires_grad) right->grad[k * columns + column] += gradient * left->data[row * inner + k];
                }
            }
        }
    };
    return TensorAccess::wrap(std::move(output));
}

Tensor Tensor::sum() const {
    const auto value = std::accumulate(data().begin(), data().end(), 0.0);
    auto parent = node_;
    auto output = make_node({value}, {}, requires_grad(), "sum", {parent});
    output->backward = [parent](const std::vector<double>& upstream) {
        if (!parent->requires_grad) return;
        for (auto& value : parent->grad) value += upstream[0];
    };
    return TensorAccess::wrap(std::move(output));
}

Tensor Tensor::mean() const {
    auto result = sum() / Tensor(static_cast<double>(size()));
    TensorAccess::node(result)->op = "mean";
    return result;
}

Tensor Tensor::pow(double exponent) const {
    std::vector<double> values(size());
    std::transform(data().begin(), data().end(), values.begin(),
                   [exponent](double value) { return std::pow(value, exponent); });
    auto parent = node_;
    auto output = make_node(std::move(values), shape(), requires_grad(), "pow", {parent});
    output->backward = [parent, exponent](const std::vector<double>& upstream) {
        if (!parent->requires_grad) return;
        for (std::size_t index = 0; index < upstream.size(); ++index) {
            parent->grad[index] += upstream[index] * exponent * std::pow(parent->data[index], exponent - 1.0);
        }
    };
    return TensorAccess::wrap(std::move(output));
}

template <typename Forward, typename Derivative>
Tensor unary_tensor(const Tensor& input, const char* name, Forward forward, Derivative derivative) {
    const auto parent = TensorAccess::node(input);
    std::vector<double> values(input.size());
    std::transform(input.data().begin(), input.data().end(), values.begin(), forward);
    const auto output_values = values;
    auto output = make_node(std::move(values), input.shape(), input.requires_grad(), name, {parent});
    output->backward = [parent, output_values, derivative](const std::vector<double>& upstream) {
        if (!parent->requires_grad) return;
        for (std::size_t index = 0; index < upstream.size(); ++index) {
            parent->grad[index] += upstream[index] * derivative(parent->data[index], output_values[index]);
        }
    };
    return TensorAccess::wrap(std::move(output));
}

Tensor Tensor::log() const {
    return unary_tensor(*this, "log", [](double x) { return std::log(x); },
                        [](double x, double) { return 1.0 / x; });
}
Tensor Tensor::exp() const {
    return unary_tensor(*this, "exp", [](double x) { return std::exp(x); },
                        [](double, double y) { return y; });
}
Tensor Tensor::sin() const {
    return unary_tensor(*this, "sin", [](double x) { return std::sin(x); },
                        [](double x, double) { return std::cos(x); });
}
Tensor Tensor::sigmoid() const {
    return unary_tensor(*this, "sigmoid", [](double x) {
        return x >= 0.0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
    }, [](double, double y) { return y * (1.0 - y); });
}
Tensor Tensor::relu() const {
    return unary_tensor(*this, "relu", [](double x) { return std::max(0.0, x); },
                        [](double x, double) { return x > 0.0 ? 1.0 : 0.0; });
}
Tensor Tensor::tanh() const {
    return unary_tensor(*this, "tanh", [](double x) { return std::tanh(x); },
                        [](double, double y) { return 1.0 - y * y; });
}

void Tensor::backward() {
    if (size() != 1) throw std::invalid_argument("a gradient is required for non-scalar outputs");
    backward({1.0});
}

void Tensor::backward(const std::vector<double>& gradient) {
    if (gradient.size() != size()) throw std::invalid_argument("gradient size does not match output");
    std::vector<NodePtr> ordering;
    std::unordered_set<detail::Node*> visited, active;
    std::function<void(const NodePtr&)> visit = [&](const NodePtr& node) {
        if (active.count(node.get())) throw std::invalid_argument("computational graph contains a cycle");
        if (visited.count(node.get())) return;
        active.insert(node.get());
        for (const auto& parent : node->parents) visit(parent);
        active.erase(node.get());
        visited.insert(node.get());
        ordering.push_back(node);
    };
    visit(node_);
    for (const auto& node : ordering) std::fill(node->grad.begin(), node->grad.end(), 0.0);
    node_->grad = gradient;
    for (auto iterator = ordering.rbegin(); iterator != ordering.rend(); ++iterator) {
        if ((*iterator)->backward) (*iterator)->backward((*iterator)->grad);
    }
}

void Tensor::zero_grad() { std::fill(node_->grad.begin(), node_->grad.end(), 0.0); }

void Tensor::step(double learning_rate) {
    if (!requires_grad()) throw std::invalid_argument("step requires requires_grad=true");
    for (std::size_t index = 0; index < size(); ++index) node_->data[index] -= learning_rate * node_->grad[index];
}

double Tensor::item() const {
    if (size() != 1) throw std::invalid_argument("item requires a one-element tensor");
    return data()[0];
}

}  // namespace autodiff

