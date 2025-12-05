#pragma once
#include <inttypes.h>
#include <memory>
#include "Storage.hpp"
#include "Enums.hpp"
#include "Registry.hpp"

class CpuStorage: public Storage {
public:

    CpuStorage(size_t elemnt_cnt, DType dtype): Storage(dtype, elemnt_cnt), data(std::make_unique<uint8_t[]>(size)) {}

   
    void fill(double value) override {
        FILL_REGISTRY.dispatch(dtype, DeviceType::CPU)(data.get(), value, numel);
    }

    double read(uint32_t offset) override {
        return READ_REGISTRY.dispatch(dtype, DeviceType::CPU)(data.get(), offset);
    }

    void write(double value, uint32_t index) override {
        WRITE_ELEM_REGISTRY.dispatch(dtype, DeviceType::CPU)(data.get(), value, index);
    }

    
    std::shared_ptr<Storage> add(std::shared_ptr<Storage> other) override {
        return ADD_REGISTRY.dispatch(dtype, DeviceType::CPU)(this, other.get()); 
    }

    std::unique_ptr<uint8_t[]> data;

};