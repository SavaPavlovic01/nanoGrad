#include "Tensor.hpp"
#include "Registry.hpp"
#include <iostream>

int main() {
    register_all_fill_kernels();
    register_all_read_kernels();

    auto tensor = Tensor::ones({3, 3}, DType::Float64);

    for(uint32_t i = 0; i < 3; i++) {
        for(uint32_t j = 0; j < 3; j++) {
            std::cout<<tensor.index({i, j})<< " ";
        }
        std::cout<<std::endl;
    }

    return 0;
}