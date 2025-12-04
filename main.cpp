#include "Tensor.hpp"
#include "Registry.hpp"
#include <iostream>

int main() {
    register_all_fill_kernels();
    register_all_read_kernels();
    auto tensor = Tensor::ones({3, 3}, DType::Float32, DeviceType::GPU);
    std::cout<<tensor.index({1, 1})<<std::endl;
    return 0;
}