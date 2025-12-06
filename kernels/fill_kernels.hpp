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




template<typename T>
void fill_kernel_gpu_better(cl_mem buffer, float val, uint64_t size) {
    auto& context = OpenCLContext::get();

    std::string kernel_name = std::format("fill_{}", OpenCLContext::type_to_cl_string<T>());

    cl_kernel kernel;
    std::optional<cl_kernel> probe_kernel = context.get_kernel_by_name(kernel_name);
    if(!probe_kernel.has_value()) {
        // really slow, in a real thing they would precompile all the kernels in a binary right?
        std::string kernel_src = std::format(R"(
            __kernel void {}(__global {}* a,
                           float val, // I DONT HAVE SUPPORT FOR DOUBLE PRECISION :(
                           ulong n) {{
                size_t idx = get_global_id(0);
                if (idx >= n) return;
                a[idx] = ({})val;
            }}
        )", kernel_name,
            OpenCLContext::type_to_cl_string<T>(),
            OpenCLContext::type_to_cl_string<T>());
        kernel = context.get_or_make_kernel(kernel_name, kernel_src);
    } else {
        kernel = probe_kernel.value();
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    clSetKernelArg(kernel, 1, sizeof(float) ,&val);
    clSetKernelArg(kernel, 2, sizeof(uint64_t), &size);
    
    size_t global_size = ((size + 255) / 256) * 256;
    context.runKernel(kernel, {global_size});
}