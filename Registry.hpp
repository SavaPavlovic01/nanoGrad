#pragma once
#include <unordered_map>
#include <utility>
#include "Enums.hpp"
#include <inttypes.h>

using FillFn = void(*)(void*, double, uint64_t);


template<typename Fn>
class Registry {
public:

    Registry() {}

    void register_kernel(DType dtype, DeviceType device, Fn kernel) {
        registry[makeKey(dtype, device)] = kernel;
    }

    Fn dispatch(DType dtype, DeviceType device) {
        return registry.at(makeKey(dtype, device));
    }

    uint64_t makeKey(DType dtype, DeviceType device){
        return (static_cast<uint64_t>(dtype)) << 32 | static_cast<uint64_t>(device);
    }
    
private:
    std::unordered_map<uint64_t, Fn> registry;
};

extern Registry<FillFn> FILL_REGISTRY;
extern void register_all_fill_kernels();

using ReadFn = double(*)(void*, uint32_t);
extern Registry<ReadFn> READ_REGISTRY;
extern void register_all_read_kernels();

