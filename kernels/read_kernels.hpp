#pragma once
#include <inttypes.h>
#include <type_traits>
#include "OpenClContext.hpp"

template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
double read_kernel_cpu(void* data, uint32_t index) {
    T* ptr = reinterpret_cast<T*>(data);
    return static_cast<double>(ptr[index]);
}


double read_kernel_gpu_openCl(void* data, uint32_t index) {
    float ret;   
    auto& context = OpenCLContext::get();
    context.readInOneFloat(reinterpret_cast<cl_mem>(data), index, &ret);
    return ret;
}