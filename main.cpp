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
    auto tensor = Tensor::rand({1}, 42, DeviceType::GPU);
    auto t1 =  Tensor::rand({1}, 42, DeviceType::GPU);
    auto t2 =  Tensor::rand({1}, 42, DeviceType::GPU);
    auto t3 =  Tensor::rand({1}, 42, DeviceType::GPU);

    auto mult = tensor * t1;
    auto t4 = mult + t2;
    if(t4.shape.empty() || t1.shape.empty() || t2.shape.empty() || tensor.shape.empty() ){
        std::cout<<"SMOEONE";
    }
    t4.backward();
    
    std::cout<<"Je;;"<<std::endl;
    std::cout<<std::endl<<t1.grad->index({0})<<std::endl;
    return 0;
}