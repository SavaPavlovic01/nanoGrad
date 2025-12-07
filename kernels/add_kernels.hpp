#pragma once
#include <CL/cl.h>
#include <string>
#include <memory>
#include "Storage.hpp"
#include "Enums.hpp"
#include "OpenClContext.hpp"
#include <format>
#include <optional>


template<typename T1, typename T2, typename Tout>
void add_kernel_cpu_better(const T1* a, const T2* b, Tout* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = static_cast<Tout>(a[i]) + static_cast<Tout>(b[i]);
}


template<typename T1, typename T2, typename Tout>
void add_kernel_opencl(cl_mem a, cl_mem b, cl_mem out,size_t n) {
    auto& context = OpenCLContext::get(); 
    std::string kernel_name = std::format("add_{}_{}_{}", OpenCLContext::type_to_cl_string<T1>(), 
        OpenCLContext::type_to_cl_string<T2>(), OpenCLContext::type_to_cl_string<Tout>());

    cl_kernel kernel;
    std::optional<cl_kernel> probe_kernel = context.get_kernel_by_name(kernel_name);
    if(!probe_kernel.has_value()) {
        // really slow, in a real thing they would precompile all the kernels in a binary right?
        std::string kernel_src = std::format(R"(
            __kernel void {}(__global const {}* a,
                           __global const {}* b,
                           __global {}* out,
                           ulong n) {{
                size_t idx = get_global_id(0);
                if (idx >= n) return;
                out[idx] = ({})a[idx] + ({})b[idx];
            }}
        )", kernel_name,
            OpenCLContext::type_to_cl_string<T1>(),
            OpenCLContext::type_to_cl_string<T2>(),
            OpenCLContext::type_to_cl_string<Tout>(),
            OpenCLContext::type_to_cl_string<Tout>(),
            OpenCLContext::type_to_cl_string<Tout>());

        kernel = context.get_or_make_kernel(kernel_name, kernel_src);
    } else {
        kernel = probe_kernel.value();
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 3, sizeof(uint64_t), &n);
    
    size_t global_size = ((n + 255) / 256) * 256;
    context.runKernel(kernel, {global_size});
}

template<typename T1>
void add_constant_kernel_opencl(cl_mem a, double value, size_t n) {
    auto& context = OpenCLContext::get(); 
    std::string kernel_name = std::format("add_constant_{}", OpenCLContext::type_to_cl_string<T1>());

    cl_kernel kernel;
    std::optional<cl_kernel> probe_kernel = context.get_kernel_by_name(kernel_name);
    if(!probe_kernel.has_value()) {
        std::string kernel_src = std::format(R"(
            __kernel void {}(__global {}* dest,
                           const {} value,
                           ulong n) {{
                size_t idx = get_global_id(0);
                if (idx >= n) return;
                dest[idx] += value;
            }}
        )", kernel_name,
            OpenCLContext::type_to_cl_string<T1>(),
            OpenCLContext::type_to_cl_string<T1>());

        kernel = context.get_or_make_kernel(kernel_name, kernel_src);
    } else {
        kernel = probe_kernel.value();
    }

    T1 add_val = static_cast<T1>(value);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
    clSetKernelArg(kernel, 1, sizeof(T1), &add_val);
    clSetKernelArg(kernel, 2, sizeof(size_t), &n);
    
    size_t global_size = ((n + 255) / 256) * 256;
    context.runKernel(kernel, {global_size});
}
