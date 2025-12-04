#pragma once
#include <inttypes.h>
#include <type_traits>

template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
double read_kernel_cpu(void* data, uint32_t index) {
    T* ptr = reinterpret_cast<T*>(data);
    return static_cast<double>(ptr[index]);
}