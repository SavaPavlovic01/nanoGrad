#include "Tensor.hpp"
#include "Registry.hpp"
#include <iostream>

int main() {
    register_all_fill_kernels();
    register_all_read_kernels();
    register_all_write_elem();
    register_all_add_kernels();
    auto tensor = Tensor::rand({3, 3}, 42, DeviceType::GPU);
    std::cout<<tensor.index({1, 1})<<std::endl;
    std::cout<<tensor.index({2, 2})<<std::endl;
    std::cout<<tensor.index({0, 0})<<std::endl;
    return 0;
}