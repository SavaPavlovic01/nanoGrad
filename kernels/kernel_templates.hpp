#pragma once
#include <CL/cl.h>
#include "OpenClContext.hpp"
#include <format>
#include <string>
#include <optional>

inline std::string get_op_name(const std::string& op) {
    if (op == "+") return "add";
    else if(op == "*") return "mult";
    else if(op == "-") return "sub";
    else if(op == "/") return "div";
    else return "";
}

template<typename T1, typename T2, typename Tout>
void binary_op_into_dest(cl_mem a, cl_mem b, cl_mem out, size_t n,const std::string& op) {
    auto& context = OpenCLContext::get(); 
    std::string kernel_name = std::format("{}_{}_{}_{}", get_op_name(op) ,OpenCLContext::type_to_cl_string<T1>(), 
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
                out[idx] = ({})a[idx] {} ({})b[idx];
            }}
        )", kernel_name,
            OpenCLContext::type_to_cl_string<T1>(),
            OpenCLContext::type_to_cl_string<T2>(),
            OpenCLContext::type_to_cl_string<Tout>(),
            OpenCLContext::type_to_cl_string<Tout>(),
            op,
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

template<typename T1, typename T2>
void binary_op_from_buffer_into_dest(cl_mem dest, cl_mem b, size_t n, const std::string& op) {
    auto& context = OpenCLContext::get(); 
    std::string kernel_name = std::format("{}_intoDest_{}_{}", get_op_name(op), OpenCLContext::type_to_cl_string<T1>(), OpenCLContext::type_to_cl_string<T2>());

    cl_kernel kernel;
    std::optional<cl_kernel> probe_kernel = context.get_kernel_by_name(kernel_name);
    if(!probe_kernel.has_value()) {
        std::string kernel_src = std::format(R"(
            __kernel void {}(__global {}* dest,
                           __global const {}* b,
                           ulong n) {{
                size_t idx = get_global_id(0);
                if (idx >= n) return;
                dest[idx] {}= ({})b[idx]; 
            }}
        )", kernel_name,
            OpenCLContext::type_to_cl_string<T1>(),
            OpenCLContext::type_to_cl_string<T2>(),
            op, OpenCLContext::type_to_cl_string<T1>());
        kernel = context.get_or_make_kernel(kernel_name, kernel_src);
    } else {
        kernel = probe_kernel.value();
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dest);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
    clSetKernelArg(kernel, 2, sizeof(uint64_t), &n);
    
    size_t global_size = ((n + 255) / 256) * 256;
    context.runKernel(kernel, {global_size});
}

template<typename T>
void binary_op_from_const_into_dest(cl_mem dest, float value, size_t n, const std::string& op) {
    auto& context = OpenCLContext::get(); 
    std::string kernel_name = std::format("{}_intoDest_{}", get_op_name(op), OpenCLContext::type_to_cl_string<T>());

    cl_kernel kernel;
    std::optional<cl_kernel> probe_kernel = context.get_kernel_by_name(kernel_name);
    if(!probe_kernel.has_value()) {
        std::string kernel_src = std::format(R"(
            __kernel void {}(__global {}* dest,
                           float b,
                           ulong n) {{
                size_t idx = get_global_id(0);
                if (idx >= n) return;
                dest[idx] {}= ({})b; 
            }}
        )", kernel_name,
            OpenCLContext::type_to_cl_string<T>(),
            op, 
            OpenCLContext::type_to_cl_string<T>());
        kernel = context.get_or_make_kernel(kernel_name, kernel_src);
    } else {
        kernel = probe_kernel.value();
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dest);
    clSetKernelArg(kernel, 1, sizeof(sizeof(float)), &value);
    clSetKernelArg(kernel, 2, sizeof(uint64_t), &n);
    
    size_t global_size = ((n + 255) / 256) * 256;
    context.runKernel(kernel, {global_size});
}
