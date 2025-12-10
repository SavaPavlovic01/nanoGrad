#pragma once
#include <CL/cl.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <optional>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>

class OpenCLContext {
public:
    static OpenCLContext& get() {
        static OpenCLContext instance;
        return instance;
    }

    cl_program getProgram(const std::string& src, const char* build_args = nullptr) {
        if(program_cache.count(src)) {
            return program_cache.at(src);
        }
        cl_int err;
        const char* source = src.c_str();
        cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL program");
        }

        err = clBuildProgram(program, 1, &device, build_args, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "OpenCL build error:\n" << log.data() << std::endl;
            throw std::runtime_error("Failed to build OpenCL program");
        }

        program_cache[source] = program;

        cl_uint kernelCnt = 0;
        err = clCreateKernelsInProgram(program, 0, nullptr, &kernelCnt);
        if (err != CL_SUCCESS)
            throw std::runtime_error("Failed clCreateKernelsInProgram");
        std::vector<cl_kernel> kernels(kernelCnt);

        clCreateKernelsInProgram(program, kernelCnt, kernels.data(), nullptr);

        for (auto &kernel : kernels){
            char name[128];
            size_t sz;
            clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(name), name, &sz);
            kernel_cache[name] = kernel;
        }

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

    // use only if you know the kernel exists
    std::optional<cl_kernel> get_kernel_by_name(const std::string& name) {
        if(kernel_cache.contains(name)) return kernel_cache[name];
        return std::nullopt;
    }

    cl_kernel get_or_make_kernel(const std::string& name, const std::string& source) {
        if(kernel_cache.contains(name)) return kernel_cache[name];
        cl_program program = getProgram(source);
        return kernel_cache[name];
    }

    cl_mem allocateBuffer(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE) {
        cl_int err;
        cl_mem buf = clCreateBuffer(context, flags, size, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL buffer");
        }
        return buf;
    }

    // returns kernel run time
    double runKernel(cl_kernel kernel,
                   const std::vector<size_t> &global_work_size,
                   const std::vector<size_t> &local_work_size = {}){
        cl_int err;
        cl_event event = NULL;
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
            &event);

        if (err != CL_SUCCESS)
            throw std::runtime_error("Failed to enqueue OpenCL kernel");
        clFinish(queue);

        cl_ulong time_start, time_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,   sizeof(time_end),   &time_end,   NULL);
    
        double elapsed_ns = (double)(time_end - time_start);
        double elapsed_ms = elapsed_ns * 1e-6;
        std::cout<<"Runtime "<<elapsed_ms<<std::endl;
        return elapsed_ms;
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

    template <typename T>
    static const char *type_to_cl_string(){
        if constexpr (std::is_same_v<T, float>)
            return "float";
        else if constexpr (std::is_same_v<T, double>)
            return "double";
        else if constexpr (std::is_same_v<T, int32_t>)
            return "int";
        else if constexpr (std::is_same_v<T, int64_t>)
            return "long";
        else if constexpr (std::is_same_v<T, uint32_t>)
            return "uint";
        else if constexpr (std::is_same_v<T, uint64_t>)
            return "ulong";
        else
            return "";
    }

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

private:

    void preloadKernels() {
        std::ifstream file("mm.cl");
        if(!file.is_open()) throw std::runtime_error("Failed to open kernel file");
    
        std::ostringstream oss;
        oss << file.rdbuf();
        std::string source = oss.str();

        getProgram(source);
    }

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

        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, nullptr);

        preloadKernels();
    }

    OpenCLContext(const OpenCLContext&) = delete;
    OpenCLContext& operator=(const OpenCLContext&) = delete;
    std::unordered_map<std::string, cl_program> program_cache;
    std::unordered_map<std::string, cl_kernel> kernel_cache;
    
};