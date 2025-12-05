#pragma once
#include "Storage.hpp"
#include "Enums.hpp"
#include <CL/cl.h>
#include "OpenClContext.hpp"
#include "Registry.hpp"
class GPUStorage : public Storage {
public:
    GPUStorage(size_t elemnt_cnt, DType dtype) : Storage(dtype, elemnt_cnt) {
        data = OpenCLContext::get().allocateBuffer(size, CL_MEM_READ_WRITE);
    }

    void fill(double value) override {
        FILL_REGISTRY.dispatch(dtype, DeviceType::GPU)(&data, value, numel);
    }

    double read(uint32_t offset) override{
        return READ_REGISTRY.dispatch(dtype, DeviceType::GPU)(data, offset); 
    }


    void write(double value, uint32_t index) override {
        WRITE_ELEM_REGISTRY.dispatch(dtype, DeviceType::GPU)(data, value, index);
    }

    
    std::shared_ptr<Storage> add(std::shared_ptr<Storage> other) override {
        return ADD_REGISTRY.dispatch(dtype, DeviceType::GPU)(this, other.get()); 
    }


    cl_mem data;
};