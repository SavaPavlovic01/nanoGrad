#pragma once
#include <inttypes.h>
#include "Enums.hpp"
#include <stdexcept>

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

    virtual void add_into(const std::shared_ptr<Storage>&){throw  std::runtime_error("Not yet implemented");}
    virtual void div_into(const std::shared_ptr<Storage>&){throw  std::runtime_error("Not yet implemented");}
    virtual void mult_into(const std::shared_ptr<Storage>&){throw  std::runtime_error("Not yet implemented");}
    virtual void sub_into(const std::shared_ptr<Storage>&){throw  std::runtime_error("Not yet implemented");}

    virtual void add_into(double value){throw  std::runtime_error("Not yet implemented");}
    virtual void div_into(double value){throw  std::runtime_error("Not yet implemented");}
    virtual void mult_into(double value){throw  std::runtime_error("Not yet implemented");}
    virtual void sub_into(double value){throw  std::runtime_error("Not yet implemented");}

    virtual void negate(){throw std::runtime_error("Not yet implemented");}

    virtual std::shared_ptr<Storage> tanh(const std::vector<uint32_t> shape, const std::vector<uint32_t> strides, size_t numel){ throw std::runtime_error("Not implemented");}

    virtual std::shared_ptr<Storage> contiguous( std::vector<uint32_t>& shape,  std::vector<uint32_t>& strides, uint32_t ndim, uint32_t numel) {throw  std::runtime_error("Not yet implemented");}

    virtual std::shared_ptr<Storage>mm(const std::shared_ptr<Storage>& other,const std::vector<uint32_t>& other_sizes, const std::vector<uint32_t>& other_strides, 
        const std::vector<uint32_t>& this_sizes, const std::vector<uint32_t>& this_strides){throw std::runtime_error("not yet");}

    virtual std::shared_ptr<Storage> softmax(const std::vector<uint32_t>& shape, const std::vector<uint32_t> stride, size_t numel){throw std::runtime_error("not yet");};
    virtual std::shared_ptr<Storage> cross_entropy(const std::shared_ptr<Storage>& targets, const std::vector<uint32_t> shape) {throw std::runtime_error("not yet");}

    virtual void write(double value, uint32_t index) = 0;
    uint64_t get_numel() { return numel;}
    virtual void rand_fill(uint32_t seed) = 0;
    
    uint64_t numel;
    DType dtype;
    uint64_t size;
};