#include "Tensor.hpp"
#include "Registry.hpp"
#include <iostream>

int main() {
    register_all_fill_kernels();
    register_all_read_kernels();
    register_all_write_elem();
    register_all_add_kernels();
    uint32_t size = 2000;
    auto tensor = Tensor::rand({size, size}, 42, DeviceType::GPU);
    auto tensor1 = Tensor::rand({size, size}, 42, DeviceType::GPU);

    auto res = tensor.mm(tensor1);
    return 0;
}