#include "Registry.hpp"
#include "add_kernels.hpp"
#include "Enums.hpp"
#include "fill_kernels.hpp"
#include "read_kernels.hpp"
#include "write_kernel.hpp"

Registry<FillFn> FILL_REGISTRY;
Registry<ReadFn> READ_REGISTRY;
Registry<WriteElementFn> WRITE_ELEM_REGISTRY;
Registry<AddFn> ADD_REGISTRY;

void register_all_fill_kernels() {
    FILL_REGISTRY.register_kernel(DType::Float32, DeviceType::CPU, &fill_kernel_cpu<float>);
    FILL_REGISTRY.register_kernel(DType::Float64, DeviceType::CPU, &fill_kernel_cpu<double>);
    FILL_REGISTRY.register_kernel(DType::Int32, DeviceType::CPU, &fill_kernel_cpu<int32_t>);
    FILL_REGISTRY.register_kernel(DType::Int64, DeviceType::CPU, &fill_kernel_cpu<int64_t>);

    //FILL_REGISTRY.register_kernel(DType::Float32, DeviceType::GPU, &fill_kernel_gpu_openCl); 
}

void register_all_read_kernels() {
    READ_REGISTRY.register_kernel(DType::Float32, DeviceType::CPU, &read_kernel_cpu<float>);
    READ_REGISTRY.register_kernel(DType::Float64, DeviceType::CPU, &read_kernel_cpu<double>);
    READ_REGISTRY.register_kernel(DType::Int32, DeviceType::CPU, &read_kernel_cpu<int32_t>);
    READ_REGISTRY.register_kernel(DType::Int64, DeviceType::CPU, &read_kernel_cpu<int64_t>);

    READ_REGISTRY.register_kernel(DType::Float32, DeviceType::GPU, &read_kernel_gpu_openCl);
}

void register_all_write_elem() {
    WRITE_ELEM_REGISTRY.register_kernel(DType::Float32, DeviceType::CPU, &write_one_element_cpu<float>);
    WRITE_ELEM_REGISTRY.register_kernel(DType::Float64, DeviceType::CPU, &write_one_element_cpu<double>);
    WRITE_ELEM_REGISTRY.register_kernel(DType::Int32, DeviceType::CPU, &write_one_element_cpu<int32_t>);
    WRITE_ELEM_REGISTRY.register_kernel(DType::Int64, DeviceType::CPU, &write_one_element_cpu<int64_t>);
}

void register_all_add_kernels() {
//     ADD_REGISTRY.register_kernel(DType::Float32, DeviceType::CPU, &add_kernel_cpu<float>);
//     ADD_REGISTRY.register_kernel(DType::Float64, DeviceType::CPU, &add_kernel_cpu<double>);
//     ADD_REGISTRY.register_kernel(DType::Int32, DeviceType::CPU, &add_kernel_cpu<int32_t>);
//     ADD_REGISTRY.register_kernel(DType::Int64, DeviceType::CPU, &add_kernel_cpu<int64_t>);
// 
   // ADD_REGISTRY.register_kernel(DType::Float32, DeviceType::GPU, &add_kernel_gpu_f32);
// 
} 