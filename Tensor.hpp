#pragma once

#include <vector>
#include <memory>
#include "Enums.hpp"

class Storage;
class GradFn;

class Tensor {
public:
    Tensor(std::vector<uint32_t> sizes,
           DType dtype,
           DeviceType device = DeviceType::CPU);

    Tensor(std::vector<uint32_t> sizes,
           DType dtype,
           DeviceType device,
           std::shared_ptr<Storage> storage);

    static Tensor ones(std::vector<uint32_t> shape,
                       DType dtype = DType::Float32,
                       DeviceType device = DeviceType::CPU);


    static Tensor zeros(std::vector<uint32_t> shape,
                       DType dtype = DType::Float32,
                       DeviceType device = DeviceType::CPU);

    static Tensor rand(std::vector<uint32_t> sizes,
                       uint32_t seed = 42,
                       DeviceType device = DeviceType::CPU);

    float index(std::vector<uint32_t> indices) const ;

    Tensor operator+(const Tensor& b);
    Tensor operator*(const Tensor& b);
    Tensor operator-(const Tensor& b);
    Tensor operator/(const Tensor& b);

    Tensor add_const(double value);
    Tensor& operator+=(const Tensor& b);
    Tensor& operator/=(const Tensor& b);

    Tensor& operator-();

    void backward();

    Tensor mm(const Tensor& b);
    Tensor transpose();

    friend Tensor operator+(const Tensor& t, double value);
    friend Tensor operator+(double value, const Tensor& t);

    friend Tensor operator/(const Tensor& t, const Tensor t2);

    Tensor reshape(const std::vector<uint32_t> new_shape);
    Tensor contiguous();
    bool is_contiguous();

    uint32_t calc_numel(std::vector<uint32_t> sizes);
    std::vector<uint32_t> getStrides(std::vector<uint32_t>& shape);
    void lazy_init_grads();
    Tensor tanh();
    Tensor softmax();

    Tensor& negate();

    std::shared_ptr<Storage> storage;
    std::shared_ptr<Tensor> grad;
    std::vector<uint32_t> shape;
    uint32_t numel;
    std::vector<uint32_t> strides;
    DeviceType device;
    DType dtype;
    std::shared_ptr<GradFn> gradFn;

    bool requires_grad = false;
};
