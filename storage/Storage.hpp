#pragma once
#include <inttypes.h>
#include "Enums.hpp"

class Storage {
public:
    Storage(DType dtype, uint64_t numel): dtype(dtype), numel(numel), size(numel * getDTypeSize(dtype)){}

    virtual void fill(double value) = 0;
    virtual ~Storage() {}
    virtual double read(uint32_t offset) = 0;
    virtual std::shared_ptr<Storage> add(std::shared_ptr<Storage>) = 0;
    virtual std::shared_ptr<Storage> add(double value) = 0;
    virtual std::shared_ptr<Storage> mult(const std::shared_ptr<Storage>&) = 0;
    virtual std::shared_ptr<Storage> div(const std::shared_ptr<Storage>&) = 0;
    virtual std::shared_ptr<Storage> sub(const std::shared_ptr<Storage>&) = 0;
    virtual void write(double value, uint32_t index) = 0;
    uint64_t get_numel() { return numel;}
    virtual void rand_fill(uint32_t seed) = 0;
    
    uint64_t numel;
    DType dtype;
    uint64_t size;
};