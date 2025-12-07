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

template<typename T>
void copy_kernel_gpu(cl_mem src, cl_mem dest, size_t n) {
    auto& context = OpenCLContext::get();

    std::string kernel_name = std::format("copy_{}", OpenCLContext::type_to_cl_string<T>());

    cl_kernel kernel;
    std::optional<cl_kernel> probe_kernel = context.get_kernel_by_name(kernel_name);
    if(!probe_kernel.has_value()) {
        std::string kernel_src = std::format(R"(
            __kernel void {}(__global const {}* src,
                            __global {}* dest,
                           ulong n) {{
                size_t idx = get_global_id(0);
                if (idx >= n) return;
                dest[idx] = src[idx];
            }}
        )", kernel_name,
            OpenCLContext::type_to_cl_string<T>(),
            OpenCLContext::type_to_cl_string<T>());
        kernel = context.get_or_make_kernel(kernel_name, kernel_src);
    } else {
        kernel = probe_kernel.value();
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dest);
    clSetKernelArg(kernel, 2, sizeof(uint64_t), &n);
    
    size_t global_size = ((n + 255) / 256) * 256;
    context.runKernel(kernel, {global_size});
}


inline void fill_random_gpu_philox_float32(cl_mem buffer, uint64_t size, uint32_t seed) {
    std::string kernel_src = R"(
    inline uint mulhilo(uint a, uint b, uint* hi) {
        ulong r = (ulong)a * (ulong)b;
        *hi = (uint)(r >> 32);
        return (uint)r;
    }

    inline uint4 philox_round(uint4 ctr, uint2 key) {
        uint hi0, hi1;
        uint lo0 = mulhilo(0xD2511F53, ctr.x, &hi0);
        uint lo1 = mulhilo(0xCD9E8D57, ctr.z, &hi1);

        uint4 out;
        out.x = hi1 ^ ctr.y ^ key.x;
        out.y = lo1;
        out.z = hi0 ^ ctr.w ^ key.y;
        out.w = lo0;
        return out;
    }

    uint4 philox(uint4 ctr, uint2 key) {
        for (int i = 0; i < 10; ++i) {
            ctr = philox_round(ctr, key);
            key.x += 0x9E3779B9;
            key.y += 0xBB67AE85;
        }
        return ctr;
    }

    __kernel void rand_kernel(__global float* out, uint seed, ulong n) {
        size_t id = get_global_id(0);
        if(id >= n) return;

        uint4 ctr = (uint4)(id, 0, 0, 0);
        uint2 key = (uint2)(seed, seed ^ 0xdeadbeef);

        uint4 r = philox(ctr, key);

        out[id] = (float)(r.x >> 9) * (1.0f / 8388608.0f); // uniform (0,1)
    })";
    auto& context = OpenCLContext::get();
    cl_kernel kernel = context.get_or_make_kernel("rand_kernel", kernel_src);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    clSetKernelArg(kernel, 1, sizeof(uint32_t), &seed);
    clSetKernelArg(kernel, 2, sizeof(uint64_t), &size);

    size_t global_size = ((size + 255) / 256) * 256;
    context.runKernel(kernel, {global_size});
}