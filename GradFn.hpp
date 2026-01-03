#pragma once
#include "Tensor.hpp"

class Tensor;

class GradFn {
public:
    virtual void backward(std::shared_ptr<Tensor> grad_tensor) = 0;
    virtual ~GradFn() = default;
};

class AddGradFn : public GradFn {
public:
    Tensor& a;
    Tensor& b;

    AddGradFn( Tensor& a,  Tensor& b): a(a), b(b) {}

    void backward(std::shared_ptr<Tensor> grad_tensor) override {
        if(a.requires_grad) {
            a.lazy_init_grads();
            *a.grad += *grad_tensor;
        }

        if(b.requires_grad) {
            b.lazy_init_grads();
            *b.grad += *grad_tensor;
        }
        a.backward();
        b.backward();
    }

};

class SubGradFn : public GradFn {
public:
    Tensor& a;
    Tensor& b;

    SubGradFn( Tensor& a,  Tensor& b): a(a), b(b) {
        -this->b;
    }

    void backward(std::shared_ptr<Tensor> grad_tensor) override {
        if(a.requires_grad) {
            a.lazy_init_grads();
            *a.grad += *grad_tensor;
        }

        if(b.requires_grad) {
            b.lazy_init_grads();
            *b.grad += *grad_tensor;
        }
        a.backward();
        b.backward();
    }

};

class MultGradFn : public GradFn {
public:
    Tensor& a;
    Tensor& b;

    MultGradFn( Tensor& a,  Tensor& b): a(a), b(b) {}


    void backward(std::shared_ptr<Tensor> grad_tensor) override {
        if(a.requires_grad) {
            a.lazy_init_grads();
            *a.grad += b* (*grad_tensor);
        }
        if(b.requires_grad) {
            b.lazy_init_grads();
            *b.grad += a * (*grad_tensor);
        }
        a.backward();
        b.backward();
    }
};

class DivGradFn : public GradFn {
public:
    Tensor& a;
    Tensor& b;

    DivGradFn( Tensor& a,  Tensor& b): a(a), b(b) {}

    void backward(std::shared_ptr<Tensor> grad_tensor) override {
        if(a.requires_grad) {
            a.lazy_init_grads();
            *a.grad += (*grad_tensor) / b;
        }
        if(b.requires_grad) {
            // *b.grad += -(grad_tensor * a) / (b * b);
            b.lazy_init_grads();
            Tensor tempValue = a * (*grad_tensor);
            tempValue /= b;
            tempValue /= b;
            *b.grad += -tempValue; // TODO: this could be merged into one kernel
        }
        a.backward();
        b.backward();
    }
};

class MatrixMultGradFn : public GradFn {
public:
    Tensor& a;
    Tensor& b;

    MatrixMultGradFn(Tensor& a, Tensor& b) : a(a), b(b) {}

    void backward(std::shared_ptr<Tensor> grad_tensor) override {
        if(a.requires_grad) {
            a.lazy_init_grads();
            *a.grad += grad_tensor->mm(b.transpose());
        }
        if(b.requires_grad) {
            b.lazy_init_grads();
            *b.grad += a.transpose().mm(*grad_tensor);
        }
        a.backward();
        b.backward();
    }

};

class TanhGradFn : public GradFn {
public:
    Tensor& a;

    TanhGradFn(Tensor& a) : a(a) {}

    // TODO: reuse the tanh from the forward pass
    void backward(std::shared_ptr<Tensor> grad_tensor) override {
        if(a.requires_grad) {
            a.lazy_init_grads(); 
            Tensor temp = a.tanh();
            Tensor tempRes = (temp * temp).negate() + 1;
            *a.grad += *grad_tensor * tempRes; // 1 - tanh(x)^2
        }

        a.backward();
    }

};

class CrossEntropyGradFn : public GradFn {
public:
    Tensor& a;
    Tensor& targets;

    CrossEntropyGradFn(Tensor& a, Tensor& targets): a(a), targets(targets) {}
    void backward(std::shared_ptr<Tensor> grad_tensor) override {
        if(a.requires_grad) {
            a.lazy_init_grads();
            auto temp = a.cross_entropy_backprop(targets);
            *a.grad += *grad_tensor * temp;
        }
        a.backward();
    }
};