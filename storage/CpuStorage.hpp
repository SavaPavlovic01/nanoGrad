#pragma once
#include <inttypes.h>
#include <memory>
#include "Storage.hpp"
#include "Enums.hpp"
#include "Registry.hpp"
#include "add_kernels.hpp"
#include "dispatch.hpp"

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
        CpuStorage* ptr = dynamic_cast<CpuStorage*>(other.get());
        std::shared_ptr<Storage> result;
        dispatch_type_pairs(this->dtype, other->dtype, [&]<typename T1, typename T2>() {
            using Tout = decltype(std::declval<T1>() + std::declval<T2>());
            
            auto out = std::make_shared<CpuStorage>(this->numel, promoteDtype(this->dtype, other->dtype));
            add_kernel_cpu_better<T1, T2, Tout>(this->data_as<T1>(), ptr->data_as<T2>(), out->data_as<Tout>(), this->get_numel());
            
            result = out;
        });

        return result;
    }

    std::shared_ptr<Storage> mult(const std::shared_ptr<Storage>& other) override {
        throw std::runtime_error("NOt implemented");
    }
    
    std::shared_ptr<Storage> div(const std::shared_ptr<Storage>& other) override {
        throw std::runtime_error("NOt implemented");
    }

    std::shared_ptr<Storage> sub(const std::shared_ptr<Storage>& other) override {
        throw std::runtime_error("NOt implemented");
    }

    void rand_fill(uint32_t seed) override {
        return;
    }

    template<typename T>
    T* data_as() {
        return reinterpret_cast<T*>(data.get());
    }

    std::shared_ptr<Storage> add(double value) override {
        throw std::runtime_error("Not implemented");
    }

    std::unique_ptr<uint8_t[]> data;

};