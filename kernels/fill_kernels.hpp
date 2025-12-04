#pragma once
#include <inttypes.h>
#include <iostream>
#include <string>
#include "OpenClContext.hpp"

template<typename T>
void fill_kernel_cpu(void* data, double value, uint64_t size) {
    T scalar = static_cast<T>(value);
    T* ptr = reinterpret_cast<T*>(data);
    for(int i = 0; i < size; i++) {
        ptr[i] = scalar;
    }
}


std::string kernel = R"CLC(
    __kernel void ones_kernel_f(__global float* data, int n) {
        int i = get_global_id(0);
        if (i < n) {
            data[i] = 1.0f; 
        }
    }
)CLC";

void fill_kernel_gpu_openCl(void* data, double value, uint64_t size) {
    auto &context = OpenCLContext::get();
    auto program = context.getProgram(kernel);
    auto kernelHandle = context.getKernel(program, "ones_kernel_f");
    clSetKernelArg(kernelHandle, 0, sizeof(cl_mem), (cl_mem*)data);
    int sz = static_cast<int>(size);
    clSetKernelArg(kernelHandle, 1, sizeof(int), &sz);
    context.runKernel(kernelHandle, {size});
}