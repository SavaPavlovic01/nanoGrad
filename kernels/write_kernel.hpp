#pragma once
#include <memory>
#include "Storage.hpp"

template<typename T>
void write_one_element_cpu(void* data, double value, uint32_t index) {
    T* ptr = reinterpret_cast<T*>(data);
    ptr[index] = static_cast<T>(value);
}