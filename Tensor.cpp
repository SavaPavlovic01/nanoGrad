#include "Tensor.hpp"

#include "CpuStorage.hpp"
#include "GPUStorage.hpp"
#include <stdexcept>
#include "GradFn.hpp"
#include <algorithm>

Tensor::Tensor(std::vector<uint32_t> sizes,
               DType dtype,
               DeviceType device)
    : shape(sizes), dtype(dtype), device(device){
    strides = getStrides(sizes);

    uint64_t n = 1;
    for (auto size : sizes) n *= size;

    if (device == DeviceType::CPU)
        storage = std::make_shared<CpuStorage>(n, dtype);
    else
        storage = std::make_shared<GPUStorage>(n, dtype);

    numel = calc_numel(sizes);
}

Tensor::Tensor(std::vector<uint32_t> sizes,
               DType dtype,
               DeviceType device,
               std::shared_ptr<Storage> storage)
    : shape(sizes), dtype(dtype), device(device), storage(storage){
    strides = getStrides(sizes);
    numel = calc_numel(sizes);
}

Tensor Tensor::ones(std::vector<uint32_t> shape,
                    DType dtype,
                    DeviceType device){
    Tensor t(shape, dtype, device);
    t.storage->fill(1);
    return t;
}

Tensor Tensor::zeros(std::vector<uint32_t> shape,
                    DType dtype,
                    DeviceType device){
    Tensor t(shape, dtype, device);
    t.storage->fill(0);
    return t;
}

Tensor Tensor::rand(std::vector<uint32_t> sizes,
                    uint32_t seed,
                    DeviceType device){
    Tensor t(sizes, DType::Float32, device);
    t.storage->rand_fill(seed);
    return t;
}

float Tensor::index(std::vector<uint32_t> indices) const {
    uint32_t buffer_index = 0;
    for (size_t i = 0; i < indices.size(); i++)
        buffer_index += strides[i] * indices[i];

    return storage->read(buffer_index);
}

Tensor Tensor::operator+(const Tensor& b){
    auto target_dtype = promoteDtype(dtype, b.dtype);
    Tensor out_tensor(shape, target_dtype, device, storage->add(b.storage));
    if(requires_grad || b.requires_grad) {
        out_tensor.requires_grad = true;
        out_tensor.gradFn = std::make_shared<AddGradFn>(*this, const_cast<Tensor&>(b));
    }
   

    return out_tensor;
}

Tensor Tensor::operator*(const Tensor& b){
    auto target_dtype = promoteDtype(dtype, b.dtype);
    Tensor out_tensor(shape, target_dtype, device, storage->mult(b.storage));
    if(requires_grad || b.requires_grad) {
        out_tensor.requires_grad = true;
        out_tensor.gradFn = std::make_shared<MultGradFn>(*this, const_cast<Tensor&>(b));
    }


    return out_tensor;
}

Tensor Tensor::operator-(const Tensor& b){
    auto target_dtype = promoteDtype(dtype, b.dtype);

    Tensor out_tensor(shape, target_dtype, device, storage->sub(b.storage));
    if(requires_grad || b.requires_grad) {
        out_tensor.requires_grad = true;
        out_tensor.gradFn = std::make_shared<SubGradFn>(*this, const_cast<Tensor&>(b));
    }

    return out_tensor;
}

Tensor Tensor::operator/(const Tensor& b){
    auto target_dtype = promoteDtype(dtype, b.dtype);

    Tensor out_tensor(shape, target_dtype, device, storage->div(b.storage));
    if(requires_grad || b.requires_grad) {
        out_tensor.requires_grad = true;
        out_tensor.gradFn = std::make_shared<DivGradFn>(*this, const_cast<Tensor&>(b));
    }

    return out_tensor;
}

Tensor operator/(const Tensor& t, const Tensor t1) {
   auto target_dtype = promoteDtype(t.dtype, t1.dtype); 

    Tensor out_tensor(t.shape, target_dtype, t.device, t.storage->mult(t1.storage));
    if(t.requires_grad || t1.requires_grad) {
        out_tensor.requires_grad = true;
        out_tensor.gradFn = std::make_shared<DivGradFn>(const_cast<Tensor&>(t), const_cast<Tensor&>(t1));
    }

    return out_tensor;
}

Tensor Tensor::add_const(double value){
    return Tensor(shape, dtype, device, storage->add(value));
}

Tensor& Tensor::operator+=(const Tensor& b){
    storage->add_into(b.storage);
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& b){
    storage->div_into(b.storage);
    return *this;
}

Tensor& Tensor::operator-() {
    storage->negate();
    return *this;
}

Tensor Tensor::mm(const Tensor& b){
    Tensor out({shape[0], b.shape[1]}, dtype, device, storage->mm(b.storage, b.shape, b.strides, shape, strides));
    if(this->requires_grad || b.requires_grad) {
        out.requires_grad = true;
        out.gradFn = std::make_shared<MatrixMultGradFn>(*this, const_cast<Tensor&>(b));
    }

    return out;
}

Tensor operator+(const Tensor& t, double value){
    return Tensor(t.shape, t.dtype, t.device, t.storage->add(value));
}

Tensor operator+(double value, const Tensor& t){
    return t + value;
}

Tensor Tensor::reshape(const std::vector<uint32_t> new_shape){
    if (numel != calc_numel(new_shape))
        throw std::runtime_error("Cant reshape like that");

    if (!is_contiguous())
        return contiguous().reshape(new_shape);

    return Tensor(new_shape, dtype, device, storage);
}

Tensor Tensor::contiguous(){
    if (is_contiguous())
        return *this;

    return Tensor(
        shape,
        dtype,
        device,
        storage->contiguous(shape, strides, shape.size(), numel)
    );
}

bool Tensor::is_contiguous(){
    uint32_t target_stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; i--) {
        if (shape[i] != 1 && strides[i] != target_stride)
            return false;
        target_stride *= shape[i];
    }
    return true;
}

uint32_t Tensor::calc_numel(std::vector<uint32_t> sizes){
    uint32_t n = 1;
    for (auto s : sizes) n *= s;
    return n;
}

std::vector<uint32_t> Tensor::getStrides(std::vector<uint32_t>& shape){
    uint64_t acc = 1;
    std::vector<uint32_t> strides(shape.size());

    for (int i = (int)shape.size() - 1; i >= 0; i--) {
        strides[i] = acc;
        acc *= shape[i];
    }
    return strides;
}

void Tensor::backward() {
    if(!requires_grad || gradFn.get() == nullptr) {
        return;
    }
    if(grad.get() == nullptr) {
        this->grad = std::make_shared<Tensor>(Tensor::ones(shape, dtype, device));
    }
    gradFn->backward(grad);
}

void Tensor::lazy_init_grads() {
    if(requires_grad && grad.get() == nullptr) {
        grad = std::make_shared<Tensor>(Tensor::zeros(shape, dtype, device));
    }
    return;
}

Tensor Tensor::transpose() {
    Tensor out(shape, dtype, device, storage);
    std::reverse(out.shape.begin(), out.shape.end());
    std::reverse(out.strides.begin(), out.strides.end());
    return out;
}

// output tensor is always float32
// TODO: add faster tanh kernel and check here if the tensor is contiguous
Tensor Tensor::tanh(){
    auto out = Tensor(shape, dtype, device, storage->tanh(shape, strides, numel));
    if(requires_grad) {
        out.requires_grad = true;
        out.gradFn = std::make_shared<TanhGradFn>(*this);
    }
    return out;
}

Tensor& Tensor::negate() {
    storage->negate();
    return *this;
}

Tensor Tensor::softmax() {
    auto out = Tensor(shape, DType::Float32, device, storage->softmax(shape, strides, numel));
    return out;
}

// TODO: not really correct, you return loss per batch, should be one number
Tensor Tensor::cross_entropy(Tensor& targets){
    auto out = Tensor({shape[0]}, DType::Float32, device, storage->cross_entropy(targets.storage, shape));
    if(requires_grad) {
        out.requires_grad = true;
        out.gradFn = std::make_shared<CrossEntropyGradFn>(*this, targets);
    }
    return out;
}

Tensor Tensor::cross_entropy_backprop(Tensor& targets) {
    auto out = Tensor(shape, dtype, device, storage->cross_entropy_backprop(targets.storage, shape));
    return out;
}