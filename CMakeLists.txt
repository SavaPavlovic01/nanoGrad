if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

cmake_minimum_required(VERSION 3.16)
project(nanoGrad)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL CONFIG REQUIRED)

set(SOURCES
    main.cpp
    Registry.cpp
    # If you move implementations to .cpp
    # storage/CpuStorage.cpp
    # OpenClContext.cpp
)

include_directories(
    ${PROJECT_SOURCE_DIR}       # For root headers (Tensor.hpp, Enums.hpp)
    ${PROJECT_SOURCE_DIR}/storage
    ${PROJECT_SOURCE_DIR}/kernels
)

# Add executable
add_executable(nanoGrad ${SOURCES})


