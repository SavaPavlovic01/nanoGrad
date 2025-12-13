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

    AddGradFn( Tensor& a,  Tensor& b): a(a), b(b) {

        if(this->a.shape.empty()) {
            std::cout<<"EMPTY\n";
        }
        if(this->b.shape.empty()) {
            std::cout<<"EMPTY\n";
        }
    }

    void backward(std::shared_ptr<Tensor> grad_tensor) override {
        std::cout<<"add"<<std::endl;
        std::cout<<grad_tensor->shape[0]<< ","<<grad_tensor->index({0})<<", "<<std::endl;
        if(a.requires_grad) {
            std::cout<<a.shape[0]<<std::endl;
            a.lazy_init_grads();
            *a.grad += *grad_tensor;
        }

        if(b.requires_grad) {
            b.lazy_init_grads();
            *b.grad += *grad_tensor;
        }
        std::cout<<"DONE WITH HIM\n";
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

    MultGradFn( Tensor& a,  Tensor& b): a(a), b(b) {
        if(a.shape.empty()) {
            std::cout<<"EMPTY\n";
        }
        if(b.shape.empty()) {
            std::cout<<"EMPTY\n";
        }
    }


    void backward(std::shared_ptr<Tensor> grad_tensor) override {
        std::cout<<"MULT"<<std::endl;
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