#pragma once
#include "Enums.hpp"
#include <inttypes.h>
#include <string>

template<typename T1>
void dispatch_second_type(DType dt2, auto&& callback) {
    switch(dt2) {
        case DType::Float32: callback.template operator()<T1, float>(); break;
        case DType::Float64: callback.template operator()<T1, double>(); break;
        case DType::Int32: callback.template operator()<T1, int32_t>(); break;
        case DType::Int64: callback.template operator()<T1, int64_t>(); break;
        default: throw std::runtime_error("Unsupported dtype");
    }
}

template<typename Func>
void dispatch_type_pairs(DType dt1, DType dt2, Func&& callback) {
    switch(dt1) {
        case DType::Float32: dispatch_second_type<float>(dt2, callback); break;
        case DType::Float64: dispatch_second_type<double>(dt2, callback); break;
        case DType::Int32: dispatch_second_type<int32_t>(dt2, callback); break;
        case DType::Int64: dispatch_second_type<int64_t>(dt2, callback); break;
        default: throw std::runtime_error("Unsupported dtype");
    }
}

template<typename Func>
void dispatch_type(DType dtype, Func&& callback) {
    switch(dtype) {
        case DType::Float32: callback.template operator()<float>(); break;
        case DType::Float64: callback.template operator()<double>(); break;
        case DType::Int32: callback.template operator()<int32_t>(); break;
        case DType::Int64: callback.template operator()<int64_t>(); break;
        default: throw std::runtime_error("Unsupported dtype");
    }
}



// Same thing just macros, i think that templates are cleaner
#define DISPATCH_ALL_NUMERIC_TYPES(DTYPE, NAME, ...) \
    switch(DTYPE) { \
        case DType::Float32: { using scalar_t = float; __VA_ARGS__; break; } \
        case DType::Float64: { using scalar_t = double; __VA_ARGS__; break; } \
        case DType::Int32:   { using scalar_t = Int32_t; __VA_ARGS__; break; } \
        case DType::Int64:   { using scalar_t = Int64_t; __VA_ARGS__; break; } \
        default: throw std::runtime_error("unsupported dtype in " NAME); \
    }

#define DISPATCH_ALL_NUMERIC_TYPE_PAIRS(T1, T2, NAME, ...) \
    if (T1 == DType::Float32 && T2 == DType::Float32) { using t1_t = float; using t2_t = float; __VA_ARGS__; } \
    else if (T1 == DType::Float32 && T2 == DType::Float64) { using t1_t = float; using t2_t = double; __VA_ARGS__; } \
    else if (T1 == DType::Float64 && T2 == DType::Float32) { using t1_t = double; using t2_t = float; __VA_ARGS__; } \
    else if (T1 == DType::Float64 && T2 == DType::Float64) { using t1_t = double; using t2_t = double; __VA_ARGS__; } \
    /* add Int/Int/Int64/etc */ \
    else throw std::runtime_error("unsupported dtype pair in " NAME);