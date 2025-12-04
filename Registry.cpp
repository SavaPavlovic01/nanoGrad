#include "Registry.hpp"
#include "Enums.hpp"
#include "fill_kernels.hpp"
#include "read_kernels.hpp"

Registry<FillFn> FILL_REGISTRY;
Registry<ReadFn> READ_REGISTRY;

void register_all_fill_kernels() {
    FILL_REGISTRY.register_kernel(DType::Float32, DeviceType::CPU, &fill_kernel_cpu<float>);
    FILL_REGISTRY.register_kernel(DType::Float64, DeviceType::CPU, &fill_kernel_cpu<double>);
    FILL_REGISTRY.register_kernel(DType::Int32, DeviceType::CPU, &fill_kernel_cpu<int32_t>);
    FILL_REGISTRY.register_kernel(DType::Int64, DeviceType::CPU, &fill_kernel_cpu<int64_t>);

    FILL_REGISTRY.register_kernel(DType::Float32, DeviceType::GPU, &fill_kernel_gpu_openCl); 
}

void register_all_read_kernels() {
    READ_REGISTRY.register_kernel(DType::Float32, DeviceType::CPU, &read_kernel_cpu<float>);
    READ_REGISTRY.register_kernel(DType::Float64, DeviceType::CPU, &read_kernel_cpu<double>);
    READ_REGISTRY.register_kernel(DType::Int32, DeviceType::CPU, &read_kernel_cpu<int32_t>);
    READ_REGISTRY.register_kernel(DType::Int64, DeviceType::CPU, &read_kernel_cpu<int64_t>);

    READ_REGISTRY.register_kernel(DType::Float32, DeviceType::GPU, &read_kernel_gpu_openCl);
}