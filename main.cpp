#include "Tensor.hpp"
#include "Registry.hpp"
#include <iostream>

int main() {
    register_all_fill_kernels();
    register_all_read_kernels();
    register_all_write_elem();
    register_all_add_kernels();
    auto tensor = Tensor::ones({3, 3}, DType::Float32, DeviceType::GPU);
    auto tensor1 = Tensor::ones({3, 3}, DType::Float32, DeviceType::GPU);
    auto added = tensor + tensor1; 
    std::cout<<added.index({1, 1})<<std::endl;
    return 0;
}