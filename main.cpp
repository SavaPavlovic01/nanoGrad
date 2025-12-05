#include "Tensor.hpp"
#include "Registry.hpp"
#include <iostream>

int main() {
    register_all_fill_kernels();
    register_all_read_kernels();
    register_all_write_elem();
    register_all_add_kernels();

    auto tensor = Tensor::ones({3, 3}, DType::Float64, DeviceType::CPU);
    auto t2 = Tensor::ones({3, 3}, DType::Float32, DeviceType::CPU);
    auto res = tensor + t2;
    std::cout<<res.index({1, 1})<<std::endl;
    return 0;
}