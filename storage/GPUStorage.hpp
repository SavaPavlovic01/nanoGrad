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

    void fill(double value) override {
        dispatch_type(this->dtype, [&]<typename T>() {
            fill_kernel_gpu_better<T>(this->data, 1.0, this->numel);
        });
    }

    double read(uint32_t offset) override{
        return READ_REGISTRY.dispatch(dtype, DeviceType::GPU)(data, offset); 
    }


    void write(double value, uint32_t index) override {
        WRITE_ELEM_REGISTRY.dispatch(dtype, DeviceType::GPU)(data, value, index);
    }

    
    std::shared_ptr<Storage> add(std::shared_ptr<Storage> other) override { 
        DISPATCH_BINARY_OP(this->data, other_ptr->data, out->data, this->numel, "+");
    }

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


    cl_mem data;
};