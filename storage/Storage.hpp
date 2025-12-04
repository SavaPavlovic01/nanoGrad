#pragma once
#include <inttypes.h>
#include "Enums.hpp"

class Storage {
public:
    Storage(DType dtype, uint64_t size): dtype(dtype), size(size) {}

    virtual void fill(double value) = 0;
    virtual ~Storage() {}
    virtual double read(uint32_t offset) = 0;

protected:
    DType dtype;
    uint64_t size;
};