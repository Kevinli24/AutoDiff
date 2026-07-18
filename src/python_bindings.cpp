#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "autodiff/tensor.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ad = autodiff;

typedef struct {
    PyObject_HEAD
    ad::Tensor* tensor;
} PyTensor;

static PyTypeObject TensorType = {PyVarObject_HEAD_INIT(nullptr, 0)};

static PyObject* fail(const std::exception& error) {
    PyErr_SetString(PyExc_ValueError, error.what());
    return nullptr;
}

static bool parse_data(PyObject* object, std::vector<double>& values,
                       std::vector<std::size_t>& shape) {
    if (PyNumber_Check(object) && !PySequence_Check(object)) {
        const double value = PyFloat_AsDouble(object);
        if (PyErr_Occurred()) return false;
        values.push_back(value);
        shape.clear();
        return true;
    }

    PyObject* sequence = PySequence_Fast(object, "tensor data must be numeric or a regular nested sequence");
    if (!sequence) {
        PyErr_Clear();
        const double value = PyFloat_AsDouble(object);
        if (PyErr_Occurred()) return false;
        values.push_back(value);
        shape.clear();
        return true;
    }

    const Py_ssize_t length = PySequence_Fast_GET_SIZE(sequence);
    if (length <= 0) {
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_ValueError, "tensor dimensions must be non-empty");
        return false;
    }

    std::vector<std::size_t> child_shape;
    for (Py_ssize_t index = 0; index < length; ++index) {
        std::vector<std::size_t> current_shape;
        if (!parse_data(PySequence_Fast_GET_ITEM(sequence, index), values, current_shape)) {
            Py_DECREF(sequence);
            return false;
        }
        if (index == 0) child_shape = current_shape;
        else if (current_shape != child_shape) {
            Py_DECREF(sequence);
            PyErr_SetString(PyExc_ValueError, "nested tensor data must have a regular shape");
            return false;
        }
    }
    Py_DECREF(sequence);
    shape.clear();
    shape.push_back(static_cast<std::size_t>(length));
    shape.insert(shape.end(), child_shape.begin(), child_shape.end());
    return true;
}

static PyObject* nested_values(const std::vector<double>& values,
                               const std::vector<std::size_t>& shape,
                               std::size_t depth, std::size_t& offset) {
    if (depth == shape.size()) return PyFloat_FromDouble(values[offset++]);
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(shape[depth]));
    if (!list) return nullptr;
    for (std::size_t index = 0; index < shape[depth]; ++index) {
        PyObject* item = nested_values(values, shape, depth + 1, offset);
        if (!item) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(index), item);
    }
    return list;
}

static PyObject* wrap(ad::Tensor value) {
    PyTensor* object = reinterpret_cast<PyTensor*>(TensorType.tp_alloc(&TensorType, 0));
    if (!object) return nullptr;
    object->tensor = new ad::Tensor(std::move(value));
    return reinterpret_cast<PyObject*>(object);
}

static bool coerce(PyObject* object, ad::Tensor& result) {
    if (PyObject_TypeCheck(object, &TensorType)) {
        result = *reinterpret_cast<PyTensor*>(object)->tensor;
        return true;
    }
    std::vector<double> values;
    std::vector<std::size_t> shape;
    if (!parse_data(object, values, shape)) return false;
    try {
        result = shape.empty() ? ad::Tensor(values[0]) : ad::Tensor(std::move(values), std::move(shape));
        return true;
    } catch (const std::exception& error) {
        fail(error);
        return false;
    }
}

static PyObject* Tensor_new(PyTypeObject* type, PyObject*, PyObject*) {
    PyTensor* self = reinterpret_cast<PyTensor*>(type->tp_alloc(type, 0));
    if (self) self->tensor = nullptr;
    return reinterpret_cast<PyObject*>(self);
}

static int Tensor_init(PyTensor* self, PyObject* args, PyObject* kwargs) {
    PyObject* data = nullptr;
    int requires_grad = 0;
    static const char* names[] = {"data", "requires_grad", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", const_cast<char**>(names),
                                     &data, &requires_grad)) return -1;
    std::vector<double> values;
    std::vector<std::size_t> shape;
    if (!parse_data(data, values, shape)) return -1;
    try {
        delete self->tensor;
        self->tensor = shape.empty()
            ? new ad::Tensor(values[0], requires_grad != 0)
            : new ad::Tensor(std::move(values), std::move(shape), requires_grad != 0);
        return 0;
    } catch (const std::exception& error) {
        fail(error);
        return -1;
    }
}

static void Tensor_dealloc(PyTensor* self) {
    delete self->tensor;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyObject* Tensor_repr(PyTensor* self) {
    std::ostringstream stream;
    stream << "Tensor(shape=(";
    for (std::size_t index = 0; index < self->tensor->shape().size(); ++index) {
        if (index) stream << ", ";
        stream << self->tensor->shape()[index];
    }
    stream << "), requires_grad=" << (self->tensor->requires_grad() ? "True" : "False") << ")";
    return PyUnicode_FromString(stream.str().c_str());
}

template <typename Operation>
static PyObject* binary_operation(PyObject* left_object, PyObject* right_object, Operation operation) {
    ad::Tensor left, right;
    if (!coerce(left_object, left) || !coerce(right_object, right)) return nullptr;
    try { return wrap(operation(left, right)); }
    catch (const std::exception& error) { return fail(error); }
}

static PyObject* Tensor_add(PyObject* left, PyObject* right) {
    return binary_operation(left, right, [](const ad::Tensor& a, const ad::Tensor& b) { return a + b; });
}
static PyObject* Tensor_subtract(PyObject* left, PyObject* right) {
    return binary_operation(left, right, [](const ad::Tensor& a, const ad::Tensor& b) { return a - b; });
}
static PyObject* Tensor_multiply(PyObject* left, PyObject* right) {
    return binary_operation(left, right, [](const ad::Tensor& a, const ad::Tensor& b) { return a * b; });
}
static PyObject* Tensor_divide(PyObject* left, PyObject* right) {
    return binary_operation(left, right, [](const ad::Tensor& a, const ad::Tensor& b) { return a / b; });
}
static PyObject* Tensor_matmul_slot(PyObject* left, PyObject* right) {
    return binary_operation(left, right, [](const ad::Tensor& a, const ad::Tensor& b) { return a.matmul(b); });
}
static PyObject* Tensor_negative(PyObject* object) {
    try { return wrap(-*reinterpret_cast<PyTensor*>(object)->tensor); }
    catch (const std::exception& error) { return fail(error); }
}
static PyObject* Tensor_power(PyObject* object, PyObject* exponent, PyObject* modulus) {
    if (modulus != Py_None) {
        PyErr_SetString(PyExc_TypeError, "Tensor power does not accept a modulus");
        return nullptr;
    }
    const double value = PyFloat_AsDouble(exponent);
    if (PyErr_Occurred()) return nullptr;
    try { return wrap(reinterpret_cast<PyTensor*>(object)->tensor->pow(value)); }
    catch (const std::exception& error) { return fail(error); }
}

template <ad::Tensor (ad::Tensor::*Method)() const>
static PyObject* unary_method(PyTensor* self, PyObject*) {
    try { return wrap((self->tensor->*Method)()); }
    catch (const std::exception& error) { return fail(error); }
}

static PyObject* Tensor_matmul_method(PyTensor* self, PyObject* argument) {
    ad::Tensor other;
    if (!coerce(argument, other)) return nullptr;
    try { return wrap(self->tensor->matmul(other)); }
    catch (const std::exception& error) { return fail(error); }
}

static PyObject* Tensor_backward(PyTensor* self, PyObject* args) {
    PyObject* gradient = Py_None;
    if (!PyArg_ParseTuple(args, "|O", &gradient)) return nullptr;
    try {
        if (gradient == Py_None) self->tensor->backward();
        else {
            std::vector<double> values;
            std::vector<std::size_t> shape;
            if (!parse_data(gradient, values, shape)) return nullptr;
            if (shape != self->tensor->shape()) {
                PyErr_SetString(PyExc_ValueError, "gradient shape does not match output shape");
                return nullptr;
            }
            self->tensor->backward(values);
        }
        Py_RETURN_NONE;
    } catch (const std::exception& error) { return fail(error); }
}

static PyObject* Tensor_zero_grad(PyTensor* self, PyObject*) {
    self->tensor->zero_grad();
    Py_RETURN_NONE;
}
static PyObject* Tensor_step(PyTensor* self, PyObject* argument) {
    const double learning_rate = PyFloat_AsDouble(argument);
    if (PyErr_Occurred()) return nullptr;
    try { self->tensor->step(learning_rate); Py_RETURN_NONE; }
    catch (const std::exception& error) { return fail(error); }
}
static PyObject* Tensor_item(PyTensor* self, PyObject*) {
    try { return PyFloat_FromDouble(self->tensor->item()); }
    catch (const std::exception& error) { return fail(error); }
}

static PyObject* Tensor_get_data(PyTensor* self, void*) {
    std::size_t offset = 0;
    return nested_values(self->tensor->data(), self->tensor->shape(), 0, offset);
}
static PyObject* Tensor_get_grad(PyTensor* self, void*) {
    if (!self->tensor->requires_grad()) Py_RETURN_NONE;
    std::size_t offset = 0;
    return nested_values(self->tensor->grad(), self->tensor->shape(), 0, offset);
}
static PyObject* Tensor_get_shape(PyTensor* self, void*) {
    PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(self->tensor->shape().size()));
    if (!tuple) return nullptr;
    for (std::size_t index = 0; index < self->tensor->shape().size(); ++index) {
        PyTuple_SET_ITEM(tuple, static_cast<Py_ssize_t>(index),
                         PyLong_FromSize_t(self->tensor->shape()[index]));
    }
    return tuple;
}
static PyObject* Tensor_get_requires_grad(PyTensor* self, void*) {
    return PyBool_FromLong(self->tensor->requires_grad());
}
static PyObject* Tensor_get_op(PyTensor* self, void*) {
    return PyUnicode_FromString(self->tensor->op().c_str());
}
static PyObject* Tensor_get_parents(PyTensor* self, void*) {
    const auto parents = self->tensor->parents();
    PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(parents.size()));
    if (!tuple) return nullptr;
    for (std::size_t index = 0; index < parents.size(); ++index) {
        PyObject* parent = wrap(parents[index]);
        if (!parent) { Py_DECREF(tuple); return nullptr; }
        PyTuple_SET_ITEM(tuple, static_cast<Py_ssize_t>(index), parent);
    }
    return tuple;
}

static PyMethodDef Tensor_methods[] = {
    {"backward", reinterpret_cast<PyCFunction>(Tensor_backward), METH_VARARGS, "Run reverse-mode backpropagation."},
    {"zero_grad", reinterpret_cast<PyCFunction>(Tensor_zero_grad), METH_NOARGS, "Clear this tensor's gradient."},
    {"step", reinterpret_cast<PyCFunction>(Tensor_step), METH_O, "Apply a gradient-descent update."},
    {"item", reinterpret_cast<PyCFunction>(Tensor_item), METH_NOARGS, "Return the only scalar value."},
    {"matmul", reinterpret_cast<PyCFunction>(Tensor_matmul_method), METH_O, "Matrix multiplication."},
    {"sum", reinterpret_cast<PyCFunction>(unary_method<&ad::Tensor::sum>), METH_NOARGS, "Sum all elements."},
    {"mean", reinterpret_cast<PyCFunction>(unary_method<&ad::Tensor::mean>), METH_NOARGS, "Mean of all elements."},
    {"log", reinterpret_cast<PyCFunction>(unary_method<&ad::Tensor::log>), METH_NOARGS, "Elementwise natural log."},
    {"exp", reinterpret_cast<PyCFunction>(unary_method<&ad::Tensor::exp>), METH_NOARGS, "Elementwise exponential."},
    {"sin", reinterpret_cast<PyCFunction>(unary_method<&ad::Tensor::sin>), METH_NOARGS, "Elementwise sine."},
    {"sigmoid", reinterpret_cast<PyCFunction>(unary_method<&ad::Tensor::sigmoid>), METH_NOARGS, "Elementwise sigmoid."},
    {"relu", reinterpret_cast<PyCFunction>(unary_method<&ad::Tensor::relu>), METH_NOARGS, "Elementwise ReLU."},
    {"tanh", reinterpret_cast<PyCFunction>(unary_method<&ad::Tensor::tanh>), METH_NOARGS, "Elementwise tanh."},
    {nullptr, nullptr, 0, nullptr}
};

static PyGetSetDef Tensor_getset[] = {
    {const_cast<char*>("data"), reinterpret_cast<getter>(Tensor_get_data), nullptr, const_cast<char*>("Tensor values."), nullptr},
    {const_cast<char*>("grad"), reinterpret_cast<getter>(Tensor_get_grad), nullptr, const_cast<char*>("Accumulated gradient."), nullptr},
    {const_cast<char*>("shape"), reinterpret_cast<getter>(Tensor_get_shape), nullptr, const_cast<char*>("Tensor shape."), nullptr},
    {const_cast<char*>("requires_grad"), reinterpret_cast<getter>(Tensor_get_requires_grad), nullptr, const_cast<char*>("Whether gradients are tracked."), nullptr},
    {const_cast<char*>("op"), reinterpret_cast<getter>(Tensor_get_op), nullptr, const_cast<char*>("Creating operation."), nullptr},
    {const_cast<char*>("parents"), reinterpret_cast<getter>(Tensor_get_parents), nullptr, const_cast<char*>("Graph parents."), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

static PyNumberMethods Tensor_number_methods = {};

static PyObject* module_tensor(PyObject*, PyObject* args, PyObject* kwargs) {
    return PyObject_Call(reinterpret_cast<PyObject*>(&TensorType), args, kwargs);
}

static PyMethodDef module_methods[] = {
    {"tensor", reinterpret_cast<PyCFunction>(module_tensor), METH_VARARGS | METH_KEYWORDS, "Construct a Tensor."},
    {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_tensor",
    "C++ reverse-mode tensor automatic differentiation backend.",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__tensor() {
    Tensor_number_methods.nb_add = Tensor_add;
    Tensor_number_methods.nb_subtract = Tensor_subtract;
    Tensor_number_methods.nb_multiply = Tensor_multiply;
    Tensor_number_methods.nb_true_divide = Tensor_divide;
    Tensor_number_methods.nb_negative = Tensor_negative;
    Tensor_number_methods.nb_power = Tensor_power;
    Tensor_number_methods.nb_matrix_multiply = Tensor_matmul_slot;

    TensorType.tp_name = "autodiff._tensor.Tensor";
    TensorType.tp_basicsize = sizeof(PyTensor);
    TensorType.tp_dealloc = reinterpret_cast<destructor>(Tensor_dealloc);
    TensorType.tp_repr = reinterpret_cast<reprfunc>(Tensor_repr);
    TensorType.tp_as_number = &Tensor_number_methods;
    TensorType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    TensorType.tp_doc = "Dense tensor with reverse-mode automatic differentiation.";
    TensorType.tp_methods = Tensor_methods;
    TensorType.tp_getset = Tensor_getset;
    TensorType.tp_init = reinterpret_cast<initproc>(Tensor_init);
    TensorType.tp_new = Tensor_new;

    if (PyType_Ready(&TensorType) < 0) return nullptr;
    PyObject* module = PyModule_Create(&module_definition);
    if (!module) return nullptr;
    Py_INCREF(&TensorType);
    if (PyModule_AddObject(module, "Tensor", reinterpret_cast<PyObject*>(&TensorType)) < 0) {
        Py_DECREF(&TensorType);
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}
