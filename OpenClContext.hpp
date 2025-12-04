#pragma once
#include <CL/cl.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <iostream>

class OpenCLContext {
public:
    static OpenCLContext& get() {
        static OpenCLContext instance;
        return instance;
    }

    cl_program getProgram(const std::string& src) {
        if(program_cache.count(src)) {
            return program_cache.at(src);
        }
        cl_int err;
        const char* source = src.c_str();
        cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL program");
        }

        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "OpenCL build error:\n" << log.data() << std::endl;
            throw std::runtime_error("Failed to build OpenCL program");
        }

        program_cache[source] = program;
        return program;
    }

    cl_kernel getKernel(cl_program program, const std::string& kernel_name) {
        cl_int err;
        cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL kernel");
        }
        return kernel;
    }

    cl_mem allocateBuffer(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE) {
        cl_int err;
        cl_mem buf = clCreateBuffer(context, flags, size, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL buffer");
        }
        return buf;
    }

    void runKernel(cl_kernel kernel,
                   const std::vector<size_t> &global_work_size,
                   const std::vector<size_t> &local_work_size = {}){
        cl_int err;
        size_t dims = global_work_size.size();
        const size_t *local = (local_work_size.empty()) ? nullptr : local_work_size.data();
        err = clEnqueueNDRangeKernel(
            queue,
            kernel,
            dims,
            nullptr,
            global_work_size.data(),
            local,
            0,
            nullptr,
            nullptr);

        if (err != CL_SUCCESS)
            throw std::runtime_error("Failed to enqueue OpenCL kernel");
        clFinish(queue);
    }

    void readGpuBuffer(cl_mem buffer, size_t size_in_bytes_to_read, void* dst) {
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, size_in_bytes_to_read, dst, 0, nullptr, nullptr);
    }

    // PLS DONT USE THIS, JUST FOR TESTING
    void readInOneFloat(cl_mem buffer, uint64_t index, void* dst) {
        if(clEnqueueReadBuffer(queue, buffer, CL_TRUE, index * 4, 4, dst, 0, nullptr, nullptr) != CL_SUCCESS) {
            std::cout<<"Failed to read buffer";
        }
    }

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

private:
    OpenCLContext() {
        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        platform = platforms[0];

        cl_uint num_devices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
        device = devices[0];

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

        queue = clCreateCommandQueue(context, device, 0, nullptr);
    }

    OpenCLContext(const OpenCLContext&) = delete;
    OpenCLContext& operator=(const OpenCLContext&) = delete;
    std::unordered_map<std::string, cl_program> program_cache;
    
};