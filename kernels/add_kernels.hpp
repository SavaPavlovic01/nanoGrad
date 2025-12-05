#pragma once
#include <memory>
#include "Storage.hpp"
#include "Enums.hpp"
#include "CpuStorage.hpp"

template<typename T>
std::shared_ptr<Storage> add_kernel_cpu(Storage* t1, Storage* t2) {
    auto target_dtype = promoteDtype(t1->dtype, t2->dtype);
    auto target = std::make_shared<CpuStorage>(t1->get_numel(), target_dtype);
    for(int i = 0; i < t1->get_numel(); i++) {
        target->write(t1->read(i) + t2->read(i), i);
    }
    return target;
}