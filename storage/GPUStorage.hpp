#pragma once
#include "Storage.hpp"
#include "Enums.hpp"
#include <CL/cl.h>
#include "OpenClContext.hpp"
#include "Registry.hpp"
class GPUStorage : public Storage {
public:
    GPUStorage(size_t elemnt_cnt, DType dtype) : Storage(dtype, elemnt_cnt * getDTypeSize(dtype)) {
        data = OpenCLContext::get().allocateBuffer(size, CL_MEM_READ_WRITE);
    }

    void fill(double value) override {
        FILL_REGISTRY.dispatch(dtype, DeviceType::GPU)(&data, value, size / getDTypeSize(dtype));
    }

    double read(uint32_t offset) override{
        return READ_REGISTRY.dispatch(dtype, DeviceType::GPU)(data, offset); 
    }

    cl_mem data;
};