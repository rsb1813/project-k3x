cmake_minimum_required(VERSION 3.25)
project(k3x LANGUAGES CXX)

option(K3X_ENABLE_CUDA "Build the CUDA backend" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_library(k3x_runtime
    runtime/src/crc32c.cpp
    runtime/src/sha256.cpp
    runtime/src/reader.cpp
    runtime/src/ops.cpp
    runtime/src/profile.cpp
    runtime/src/backend_cpu.cpp
)
if(K3X_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit 13.3 REQUIRED)
    target_sources(k3x_runtime PRIVATE
        runtime/cuda/backend_cuda.cu
        runtime/cuda/device_memory.cu
        runtime/cuda/mxfp4.cu
    )
    target_link_libraries(k3x_runtime PUBLIC CUDA::cudart CUDA::cublasLt)
    set_target_properties(k3x_runtime PROPERTIES
        CUDA_ARCHITECTURES "120-real"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
else()
    target_sources(k3x_runtime PRIVATE runtime/src/backend_cuda_stub.cpp)
endif()
target_include_directories(k3x_runtime PUBLIC runtime/include)
if(MSVC)
    target_compile_options(k3x_runtime PUBLIC /utf-8)
endif()

add_executable(test_checksums tests/cpp/test_checksums.cpp)
target_link_libraries(test_checksums PRIVATE k3x_runtime)
add_test(NAME checksums COMMAND test_checksums)

add_executable(test_ops tests/cpp/test_ops.cpp)
target_link_libraries(test_ops PRIVATE k3x_runtime)
add_test(NAME ops COMMAND test_ops)

add_executable(test_profile tests/cpp/test_profile.cpp)
target_link_libraries(test_profile PRIVATE k3x_runtime)
add_test(NAME profile COMMAND test_profile)

add_executable(test_backend tests/cpp/test_backend.cpp)
target_link_libraries(test_backend PRIVATE k3x_runtime)
add_test(NAME backend COMMAND test_backend)

if(NOT K3X_ENABLE_CUDA)
    add_executable(test_backend_unavailable tests/cpp/test_backend_unavailable.cpp)
    target_link_libraries(test_backend_unavailable PRIVATE k3x_runtime)
    add_test(NAME backend_unavailable COMMAND test_backend_unavailable)
else()
    add_executable(test_cuda_device tests/cuda/test_cuda_device.cu)
    target_link_libraries(test_cuda_device PRIVATE k3x_runtime)
    set_target_properties(test_cuda_device PROPERTIES CUDA_ARCHITECTURES "120-real")
    add_test(NAME cuda_device COMMAND test_cuda_device)

    add_executable(test_cuda_dense tests/cuda/test_cuda_dense.cu)
    target_link_libraries(test_cuda_dense PRIVATE k3x_runtime)
    set_target_properties(test_cuda_dense PROPERTIES CUDA_ARCHITECTURES "120-real")
    add_test(NAME cuda_dense COMMAND test_cuda_dense)

    add_executable(test_cuda_mxfp4 tests/cuda/test_cuda_mxfp4.cu)
    target_link_libraries(test_cuda_mxfp4 PRIVATE k3x_runtime)
    set_target_properties(test_cuda_mxfp4 PROPERTIES CUDA_ARCHITECTURES "120-real")
    add_test(NAME cuda_mxfp4 COMMAND test_cuda_mxfp4)

    add_executable(test_cuda_memory tests/cuda/test_cuda_memory.cu)
    target_link_libraries(test_cuda_memory PRIVATE k3x_runtime)
    target_include_directories(test_cuda_memory PRIVATE runtime/cuda)
    set_target_properties(test_cuda_memory PROPERTIES CUDA_ARCHITECTURES "120-real")
    add_test(NAME cuda_memory COMMAND test_cuda_memory)
endif()

add_executable(test_reader tests/cpp/test_reader.cpp)
target_link_libraries(test_reader PRIVATE k3x_runtime)

add_executable(k3x_run runtime/src/main.cpp runtime/src/model.cpp)
target_link_libraries(k3x_run PRIVATE k3x_runtime)
