#pragma once
#include "Storage.hpp"
#include "Enums.hpp"
#include <CL/cl.h>
#include "OpenClContext.hpp"
#include "Registry.hpp"
#include "fill_kernels.hpp"
#include "dispatch.hpp"
#include "kernel_templates.hpp"
#include <type_traits>
#include <vector>
class GPUStorage : public Storage {
public:
    GPUStorage(size_t elemnt_cnt, DType dtype) : Storage(dtype, elemnt_cnt) {
        data = OpenCLContext::get().allocateBuffer(size, CL_MEM_READ_WRITE);
    }

    GPUStorage(size_t elemnt_cnt, DType dtype, cl_mem buffer) : Storage(dtype, elemnt_cnt), data(buffer) {}

    GPUStorage(const GPUStorage& storage): Storage(storage.dtype, storage.numel) {
        auto& context = OpenCLContext::get();
        data = context.allocateBuffer(size, CL_MEM_READ_WRITE);

        dispatch_type(this->dtype, [&]<typename T>() {
            copy_kernel_gpu<T>(storage.data, data, numel);
        });
    }

    ~GPUStorage() {
        clReleaseMemObject(data);
    }

    void fill(double value) override {
        dispatch_type(this->dtype, [&]<typename T>() {
            fill_kernel_gpu_better<T>(this->data, value, this->numel);
        });
    }

    double read(uint32_t offset) override{
        return READ_REGISTRY.dispatch(dtype, DeviceType::GPU)(data, offset); 
    }


    void write(double value, uint32_t index) override {
        WRITE_ELEM_REGISTRY.dispatch(dtype, DeviceType::GPU)(data, value, index);
    }

    // -------------------------------------------------------------------------
    // for tensor op tensor = someTensor (makes a new buffer)  
    std::shared_ptr<Storage> add(std::shared_ptr<Storage> other) override { 
        DISPATCH_BINARY_OP(this->data, other_ptr->data, out->data, this->numel, "+");
    }

    std::shared_ptr<Storage> mult(const std::shared_ptr<Storage>& other) override {
        DISPATCH_BINARY_OP(this->data, other_ptr->data, out->data, this->numel, "*");
    }

    std::shared_ptr<Storage> sub(const std::shared_ptr<Storage>& other) override {
        DISPATCH_BINARY_OP(this->data, other_ptr->data, out->data, this->numel, "-");
    }

    std::shared_ptr<Storage> div(const std::shared_ptr<Storage>& other) override {
        DISPATCH_BINARY_OP(this->data, other_ptr->data, out->data, this->numel, "/");
    }

    // ----------------------------------------------------------------------- 
    // for tensor op= tensor or const (doesnt make new buffer)

    void add_into(const std::shared_ptr<Storage>& other) override { 
        GPUStorage* ptr = dynamic_cast<GPUStorage*>(other.get());
        dispatch_type_pairs(this->dtype, other->dtype, [&]<typename T1, typename T2>() {
            binary_op_from_buffer_into_dest<T1, T2>(this->data, ptr->data, this->numel, "+");
        });
    }

    void mult_into(const std::shared_ptr<Storage>& other) override {
        GPUStorage* ptr = dynamic_cast<GPUStorage*>(other.get());
        dispatch_type_pairs(this->dtype, other->dtype, [&]<typename T1, typename T2>() {
            binary_op_from_buffer_into_dest<T1, T2>(this->data, ptr->data, this->numel, "*");
        });
    }

    void sub_into(const std::shared_ptr<Storage>& other) override {
        GPUStorage* ptr = dynamic_cast<GPUStorage*>(other.get());
        dispatch_type_pairs(this->dtype, other->dtype, [&]<typename T1, typename T2>() {
            binary_op_from_buffer_into_dest<T1, T2>(this->data, ptr->data, this->numel, "-");
        });
    }

    void div_into(const std::shared_ptr<Storage>& other) override {
        GPUStorage* ptr = dynamic_cast<GPUStorage*>(other.get());
        dispatch_type_pairs(this->dtype, other->dtype, [&]<typename T1, typename T2>() {
            binary_op_from_buffer_into_dest<T1, T2>(this->data, ptr->data, this->numel, "/");
        });
    }

    void add_into(double value) override { 
        dispatch_type(this->dtype, [&]<typename T1>() {
            binary_op_from_const_into_dest<T1>(this->data, value, this->numel, "+");
        });
    }

    void mult_into(double value) override {
        dispatch_type(this->dtype, [&]<typename T1>() {
            binary_op_from_const_into_dest<T1>(this->data, value, this->numel, "*");
        });
    }

    void sub_into(double value) override {
        dispatch_type(this->dtype, [&]<typename T1>() {
            binary_op_from_const_into_dest<T1>(this->data, value, this->numel, "-");
        });
    }

    void div_into(double value) override {
    dispatch_type(this->dtype, [&]<typename T1>() {
            binary_op_from_const_into_dest<T1>(this->data, value, this->numel, "/");
        });
    }

    // ------------------------------------------------------------------------------------

    void rand_fill(uint32_t seed) override {
        fill_random_gpu_philox_float32(data, numel, seed);
    }

    std::shared_ptr<Storage> add(double value) override {
        std::shared_ptr<Storage> result;
        dispatch_type(this->dtype, [&]<typename T1>() {

            auto out = std::make_shared<GPUStorage>(*this);
            add_constant_kernel_opencl<T1>(out->data, value, out->numel);

            result = out;
        });

        return result;
    }

    std::shared_ptr<Storage> contiguous( std::vector<uint32_t>& shape,  std::vector<uint32_t>& strides, uint32_t ndim, uint32_t numel) {
        auto& context = OpenCLContext::get();
        std::shared_ptr<Storage> result;
        dispatch_type(dtype, [&]<typename T>() {
            auto out = std::make_shared<GPUStorage>(numel, dtype);
            contiguous_kernel<T>(this->data, out->data, shape, strides, ndim, numel);
            result = out;
        });
        return result;
    }

    void negate() override {
        dispatch_type(dtype, [&]<typename T>() {
            negate_kernel_opencl<T>(data, numel);
        });
    }

    std::shared_ptr<Storage> mm(const std::shared_ptr<Storage>& other,const std::vector<uint32_t>& other_sizes, const std::vector<uint32_t>& other_strides, 
        const std::vector<uint32_t>& this_sizes, const std::vector<uint32_t>& this_strides){
        auto& context = OpenCLContext::get();
        auto kernel = context.get_kernel_by_name("matrixMult_simple");
        if(!kernel.has_value()) {
            throw std::runtime_error("WTF");
        }

        GPUStorage* ptr = dynamic_cast<GPUStorage*>(other.get());
        clSetKernelArg(kernel.value(), 0, sizeof(cl_mem), &this->data);
        clSetKernelArg(kernel.value(), 1, sizeof(uint32_t), &this_sizes[0]);
        clSetKernelArg(kernel.value(), 2, sizeof(uint32_t), &this_sizes[1]);
        clSetKernelArg(kernel.value(), 3, sizeof(uint32_t), &this_strides[0]);
        clSetKernelArg(kernel.value(), 4, sizeof(uint32_t), &this_strides[1]);

        clSetKernelArg(kernel.value(), 5, sizeof(cl_mem), &ptr->data);
        clSetKernelArg(kernel.value(), 6, sizeof(uint32_t), &other_sizes[0]);
        clSetKernelArg(kernel.value(), 7, sizeof(uint32_t), &other_sizes[1]);
        clSetKernelArg(kernel.value(), 8, sizeof(uint32_t), &other_strides[0]);
        clSetKernelArg(kernel.value(), 9, sizeof(uint32_t), &other_strides[1]);

        auto dest_buffer = context.allocateBuffer(this_sizes[1] * other_sizes[0] * sizeof(float), CL_MEM_READ_WRITE);

        clSetKernelArg(kernel.value(), 10, sizeof(cl_mem), &dest_buffer);
        size_t global_dim_x = ((this_sizes[0] + 256) / 256) * 256;
        size_t global_dim_y = ((other_sizes[1] + 256) / 256) * 256;
        context.runKernel(kernel.value(), {global_dim_x, global_dim_y});

        return std::make_shared<GPUStorage>(this_sizes[0] * other_sizes[1], DType::Float32, dest_buffer);
    }

    std::shared_ptr<Storage> tanh(const std::vector<uint32_t> shape, const std::vector<uint32_t> strides, size_t numel) override {
        auto& context = OpenCLContext::get();
        std::shared_ptr<Storage> result;
        cl_mem destBuffer = context.allocateBuffer(numel * sizeof(float), CL_MEM_READ_WRITE);
        dispatch_type(dtype, [&]<typename T>() {
            tanh_kernel_opencl<T>(data, destBuffer, shape, stride, numel);
            result = std::make_shared<GPUStorage>(numel, DType::Float32, destBuffer);
        });
        return result;
    }    


    cl_mem data;
};