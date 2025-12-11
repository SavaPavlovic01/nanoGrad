#include "Tensor.hpp"
#include "Registry.hpp"
#include <iostream>

int main() {
    register_all_fill_kernels();
    register_all_read_kernels();
    register_all_write_elem();
    register_all_add_kernels();
    auto tensor = Tensor::rand({5, 5}, 42, DeviceType::GPU);
    auto reshape = tensor.reshape({25});
    std::cout<<reshape.shape[0]<<std::endl;
    std::cout<<tensor.index({4, 3})<<std::endl;
    std::cout<<reshape.index({23})<<std::endl;
    return 0;
}