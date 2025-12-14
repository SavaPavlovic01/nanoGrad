#include "Tensor.hpp"
#include "Registry.hpp"
#include <iostream>

// TODO: t4 = t1 * t2 + t3 this does not work right now, since t1 * t2 makes a rvalue, when i do add i take its reference but it gets deleted after this line

int main() {
    register_all_fill_kernels();
    register_all_read_kernels();
    register_all_write_elem();
    register_all_add_kernels();
    uint32_t size = 3;
   // auto tensor = Tensor::rand({size, size}, 42, DeviceType::GPU);
   // tensor.requires_grad = true;
   // auto tensor1 = Tensor::rand({size, size}, 42, DeviceType::GPU);
   // auto res = tensor1 + tensor;
   // if(res.requires_grad) {
   //     std::cout<<"OK"<<std::endl;
   // }
   // std::cout<<res.index({0, 0})<<std::endl;
    auto t0 = Tensor::ones({3, 3}, DType::Float32, DeviceType::GPU);
    t0.requires_grad = true;
    auto t1 = Tensor::ones({3, 3}, DType::Float32, DeviceType::GPU);
    t1.requires_grad = true;

   
    auto res = t0.mm(t1);
    res.grad = std::make_shared<Tensor>(Tensor::ones({3, 3}, DType::Float32, DeviceType::GPU));
    res.backward();
    std::cout<<std::endl<<t0.grad->index({0, 0})<<std::endl;
    return 0;
}