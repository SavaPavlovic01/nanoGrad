#pragma once
#include <inttypes.h>
#include <iostream>

template<typename T>
void fill_kernel_cpu(void* data, double value, uint64_t size) {
    std::cout<<"Hello from kernel "<<value <<std::endl;
    T scalar = static_cast<T>(value);
    T* ptr = reinterpret_cast<T*>(data);
    for(int i = 0; i < size; i++) {
        ptr[i] = scalar;
    }
}