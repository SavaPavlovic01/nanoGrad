#pragma once
#include "Storage.hpp"
#include "Enums.hpp"
#include <CL/cl.h>
#include "OpenClContext.hpp"
#include "Registry.hpp"
#include "fill_kernels.hpp"
#include "dispatch.hpp"
class GPUStorage : public Storage {
public:
    GPUStorage(size_t elemnt_cnt, DType dtype) : Storage(dtype, elemnt_cnt) {
        data = OpenCLContext::get().allocateBuffer(size, CL_MEM_READ_WRITE);
    }

    GPUStorage(size_t elemnt_cnt, DType dtype, cl_mem buffer) : Storage(dtype, elemnt_cnt), data(buffer) {}

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
        GPUStorage* ptr = dynamic_cast<GPUStorage*>(other.get());
        std::shared_ptr<Storage> result;
        dispatch_type_pairs(this->dtype, other->dtype, [&]<typename T1, typename T2>() {
            using Tout = decltype(std::declval<T1>() + std::declval<T2>());

            auto out = std::make_shared<GPUStorage>(this->numel, promoteDtype(this->dtype, other->dtype));
            add_kernel_opencl<T1, T2, Tout>(this->data, ptr->data, out->data, this->numel);

            result = out;
        });

        return result;
    }


    cl_mem data;
};