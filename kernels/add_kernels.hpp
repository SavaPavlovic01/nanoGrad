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

template<typename T1>
void negate_kernel_opencl(cl_mem a, size_t n){
    auto& context = OpenCLContext::get(); 
    std::string kernel_name = std::format("negate_{}", OpenCLContext::type_to_cl_string<T1>());

    cl_kernel kernel;
    std::optional<cl_kernel> probe_kernel = context.get_kernel_by_name(kernel_name);
    if(!probe_kernel.has_value()) {
        std::string kernel_src = std::format(R"(
            __kernel void {}(__global {}* dest,
                           ulong n) {{
                size_t idx = get_global_id(0);
                if (idx >= n) return;
                dest[idx] *= -1;
            }}
        )", kernel_name,
            OpenCLContext::type_to_cl_string<T1>());
        kernel = context.get_or_make_kernel(kernel_name, kernel_src);
    } else {
        kernel = probe_kernel.value();
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
    clSetKernelArg(kernel, 1, sizeof(size_t), &n);
    
    size_t global_size = ((n + 255) / 256) * 256;
    context.runKernel(kernel, {global_size});
}

// TODO: for linear ones you dont have to do this loop stuff
// you can forbid tanh on int tensors, or make it return a float tensor
template<typename T1>
void tanh_kernel_opencl(cl_mem src, cl_mem dest, const std::vector<uint32_t>& shape, const std::vector<uint32_t>& stride, size_t n) {
    auto& context = OpenCLContext::get(); 
    std::string kernel_name = std::format("tanh_{}", OpenCLContext::type_to_cl_string<T1>());

    cl_kernel kernel;
    std::optional<cl_kernel> probe_kernel = context.get_kernel_by_name(kernel_name);
    if(!probe_kernel.has_value()) {
        std::string kernel_src = std::format(R"(
            __kernel void {}(__global float* dest,
                            __global const {}* src,
                            __global const uint* shape,
                            __global const uint* stride, 
                            ulong dimCnt,
                            ulong n) {{

                size_t gid = get_global_id(0);
                if (gid >= n) return;

                ulong offset = 1;
                ulong linear = gid;


                for(int d = dimCnt - 1; d >= 0; d--) {{
                    ulong idx = linear % shape[d];
                    linear /= shape[d];
                    offset += idx * stride[d];
                }}

                dest[offset] = tanh((float)src[offset]); // cast for int tensors
            }}
        )", kernel_name,
            OpenCLContext::type_to_cl_string<T1>());
        kernel = context.get_or_make_kernel(kernel_name, kernel_src);
    } else {
        kernel = probe_kernel.value();
    }

    cl_mem shape_buffer = context.allocateBuffer(shape.size() * sizeof(uint32_t), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, (void*)shape.data());
    cl_mem strides_buffer = context.allocateBuffer(stride.size() * sizeof(uint32_t), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, (void*)stride.data());

    int dimCnt = shape.size();
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dest);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &src);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &shape_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &strides_buffer);
    clSetKernelArg(kernel, 4, sizeof(uint64_t), &dimCnt);
    clSetKernelArg(kernel, 5, sizeof(uint64_t), &n);
    
    size_t global_size = ((n + 255) / 256) * 256;
    context.runKernel(kernel, {global_size});

    clReleaseMemObject(shape_buffer);
    clReleaseMemObject(strides_buffer);
}
