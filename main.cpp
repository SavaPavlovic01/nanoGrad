#include "Tensor.hpp"
#include "Registry.hpp"
#include <iostream>

// TODO: t4 = t1 * t2 + t3 this does not work right now, since t1 * t2 makes a rvalue, when i do add i take its reference but it gets deleted after this line

int main() {
    register_all_fill_kernels();
    register_all_read_kernels();
    register_all_write_elem();
    register_all_add_kernels();
    
    auto logits = Tensor::ones({5, 20}, DType::Float32, DeviceType::GPU);
    auto targets = Tensor::ones({5}, DType::Int32, DeviceType::GPU);

    auto losses = logits.cross_entropy(targets);

    std::cout<< losses.shape[0] << std::endl << losses.index({0}) << ", " << losses.index({1}) << std::endl;
    return 0;
}