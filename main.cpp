#include "Tensor.hpp"
#include "Registry.hpp"
#include <iostream>

int main() {
    register_all_fill_kernels();
    register_all_read_kernels();
    register_all_write_elem();
    register_all_add_kernels();
    auto tensor = Tensor::ones({3, 4}, DType::Float32, DeviceType::GPU);
    auto tensor1 = Tensor::ones({4, 3}, DType::Float32, DeviceType::GPU);
    auto test = tensor.mm(tensor1);
    std::cout<<test.shape[0]<<", "<<test.shape[1]<<std::endl;
    std::cout<<test.index({1, 1})<<std::endl;
    return 0;
}