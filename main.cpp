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
    auto t0 = Tensor::ones({4, 4}, DType::Float32, DeviceType::GPU);
    auto soft = t0.softmax();
    std::cout<< soft.index({1, 0}) << std::endl;
    return 0;
}