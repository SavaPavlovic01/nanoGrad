#pragma once
#include <inttypes.h>
#include <memory>
#include "Storage.hpp"
#include "Enums.hpp"
#include "Registry.hpp"

class CpuStorage: public Storage {
public:

    CpuStorage(size_t elemnt_cnt, DType dtype): Storage(dtype, elemnt_cnt * getDTypeSize(dtype)), data(std::make_unique<uint8_t[]>(size)) {}

   
    void fill(double value) override {
        FILL_REGISTRY.dispatch(dtype, DeviceType::CPU)(data.get(), value, size / getDTypeSize(dtype));
    }

    double read(uint32_t offset) override {
        return READ_REGISTRY.dispatch(dtype, DeviceType::CPU)(data.get(), offset);
    }

    std::unique_ptr<uint8_t[]> data;

};