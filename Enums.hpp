#pragma once

enum class DType {
    Float32, 
    Float64,
    Int32,
    Int64

};

enum class DeviceType {
    CPU, 
    GPU
};

inline size_t getDTypeSize(DType dtype) {
    switch (dtype)
    {
    case DType::Float32:
        return 4;
    case DType::Float64:
        return 8; 
    case DType::Int32:
        return 4;
    case DType::Int64:
        return 8;
    default:
        return 0;
        break;
    }
}