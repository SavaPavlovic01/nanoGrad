#pragma once
#include <vector>
#include "CpuStorage.hpp"
#include "GPUStorage.hpp"
#include <stdexcept>
#include "Enums.hpp"

class Tensor {
public:

    Tensor(std::vector<uint32_t> sizes, DType dtype, DeviceType device = DeviceType::CPU): shape(sizes), dtype(dtype), device(device) {
        this->strides = getStrides(sizes);
        uint64_t n = 1;
        for(auto size : sizes) n *= size;
        if(device == DeviceType::CPU) storage = std::make_shared<CpuStorage>(n, dtype);
        else {
            storage = std::make_shared<GPUStorage>(n, dtype);
        }
    }

    Tensor(std::vector<uint32_t> sizes, DType dtype, DeviceType device, std::shared_ptr<Storage> storage): shape(sizes), dtype(dtype), device(device), storage(storage) {
        strides = getStrides(sizes); 
    }

    static Tensor ones(std::vector<uint32_t> shape, DType dtype = DType::Float32, DeviceType device = DeviceType::CPU) {
        Tensor t(shape, dtype, device);
        t.storage->fill(1);
        return t;
    }

    // no dtype, just supporting float32 for now
    static Tensor rand(std::vector<uint32_t> sizes, uint32_t seed = 42, DeviceType device = DeviceType::CPU) {
        Tensor t(sizes, DType::Float32, device);
        t.storage->rand_fill(seed);
        return t;
    }

    float index(std::vector<uint32_t> indecies)  {
        uint32_t buffer_index = 0;
        for(int i = 0; i < indecies.size(); i++) {
            buffer_index += strides[i] * indecies[i];
        }

        return storage->read(buffer_index);
    }

    Tensor operator+(const Tensor& b) {
        auto target_dtype = promoteDtype(this->dtype, b.dtype);
        return Tensor(this->shape, target_dtype , this->device, this->storage->add(b.storage));
    }

    Tensor operator*(const Tensor& b) {
        auto target_dtye = promoteDtype(this->dtype, b.dtype);
        return Tensor(this->shape, target_dtye, this->device, this->storage->mult(b.storage));
    }

    Tensor operator-(const Tensor& b) {
        auto target_dtye = promoteDtype(this->dtype, b.dtype);
        return Tensor(this->shape, target_dtye, this->device, this->storage->sub(b.storage));
    }

    Tensor operator/(const Tensor& b) {
        auto target_dtye = promoteDtype(this->dtype, b.dtype);
        return Tensor(this->shape, target_dtye, this->device, this->storage->div(b.storage));
    }

    Tensor add_const(double value) {
        return Tensor(this->shape, this->dtype, device, this->storage->add(value));
    }

    Tensor& operator+=(const Tensor& b) {
        this->storage->add_into(b.storage);
        return *this;
    }

    Tensor mm(const Tensor& b) {
        return Tensor({this->shape[0], b.shape[1]}, this->dtype, device, storage->mm(b.storage, b.shape, b.strides, this->shape, this->strides));
    }

    friend Tensor operator+(const Tensor& t, const double value) {
        return Tensor(t.shape, t.dtype, t.device, t.storage->add(value));
    }


    friend Tensor operator+(const double value, const Tensor& t) {
        return t + value;
    }


    std::vector<uint32_t> getStrides(std::vector<uint32_t>& shape) {
        uint64_t acc = 1;
        std::vector<uint32_t> strides(shape.size());
        for(int i = strides.size() - 1; i >= 0; i--) {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    std::shared_ptr<Storage> storage;
    std::vector<uint32_t> shape;
    std::vector<uint32_t> strides;
    DeviceType device;
    DType dtype;

};
