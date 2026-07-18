#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace autodiff {

namespace detail {
struct Node;
}

class Tensor {
public:
    Tensor();
    Tensor(double value, bool requires_grad = false);
    Tensor(std::vector<double> data, std::vector<std::size_t> shape,
           bool requires_grad = false);

    const std::vector<double>& data() const;
    const std::vector<double>& grad() const;
    const std::vector<std::size_t>& shape() const;
    std::size_t size() const;
    bool requires_grad() const;
    const std::string& op() const;
    std::vector<Tensor> parents() const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;

    Tensor matmul(const Tensor& other) const;
    Tensor sum() const;
    Tensor mean() const;
    Tensor pow(double exponent) const;
    Tensor log() const;
    Tensor exp() const;
    Tensor sin() const;
    Tensor sigmoid() const;
    Tensor relu() const;
    Tensor tanh() const;

    void backward();
    void backward(const std::vector<double>& gradient);
    void zero_grad();
    void step(double learning_rate);
    double item() const;

private:
    explicit Tensor(std::shared_ptr<detail::Node> node);
    std::shared_ptr<detail::Node> node_;

    friend struct TensorAccess;
};

}  // namespace autodiff

