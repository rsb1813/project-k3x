cmake_minimum_required(VERSION 3.25)
project(k3x LANGUAGES CXX)

find_package(Threads REQUIRED)

option(K3X_ENABLE_CUDA "Build the CUDA backend" OFF)
option(K3X_ENABLE_IO_URING "Build the optional Linux io_uring reader" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_library(k3x_runtime
    runtime/src/crc32c.cpp
    runtime/src/sha256.cpp
    runtime/src/direct_io.cpp
    runtime/src/reader.cpp
    runtime/src/ops.cpp
    runtime/src/profile.cpp
    runtime/src/routing_policy.cpp
    runtime/src/runtime_profile.cpp
    runtime/src/backend_cpu.cpp
    runtime/src/expert_scheduler.cpp
    runtime/src/host_expert_store.cpp
    runtime/src/storage_slice.cpp
)
if(K3X_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit 13.3 REQUIRED)
    target_sources(k3x_runtime PRIVATE
        runtime/cuda/async_mxfp4_pipeline.cu
        runtime/cuda/backend_cuda.cu
        runtime/cuda/device_memory.cu
        runtime/cuda/mxfp4.cu
        runtime/cuda/pinned_memory.cu
        runtime/cuda/resident_weights.cu
        runtime/cuda/situ.cu
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
target_link_libraries(k3x_runtime PUBLIC Threads::Threads)
if(K3X_ENABLE_IO_URING)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "K3X_ENABLE_IO_URING requires Linux")
    endif()
    find_path(LIBURING_INCLUDE_DIR liburing.h REQUIRED)
    find_library(LIBURING_LIBRARY uring REQUIRED)
    target_compile_definitions(k3x_runtime PRIVATE K3X_ENABLE_IO_URING=1)
    target_include_directories(k3x_runtime PRIVATE ${LIBURING_INCLUDE_DIR})
    target_link_libraries(k3x_runtime PUBLIC ${LIBURING_LIBRARY})
endif()
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

add_executable(test_host_expert_store tests/cpp/test_host_expert_store.cpp)
target_link_libraries(test_host_expert_store PRIVATE k3x_runtime)
add_test(NAME host_expert_store COMMAND test_host_expert_store)

add_executable(test_runtime_profile tests/cpp/test_runtime_profile.cpp)
target_link_libraries(test_runtime_profile PRIVATE k3x_runtime)
add_test(NAME runtime_profile COMMAND test_runtime_profile)

add_executable(test_routing_policy tests/cpp/test_routing_policy.cpp)
target_link_libraries(test_routing_policy PRIVATE k3x_runtime)
add_test(NAME routing_policy COMMAND test_routing_policy)

add_executable(test_expert_scheduler tests/cpp/test_expert_scheduler.cpp)
target_link_libraries(test_expert_scheduler PRIVATE k3x_runtime)
add_test(NAME expert_scheduler COMMAND test_expert_scheduler)

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

    add_executable(test_cuda_pinned_memory tests/cuda/test_cuda_pinned_memory.cu)
    target_link_libraries(test_cuda_pinned_memory PRIVATE k3x_runtime)
    target_include_directories(test_cuda_pinned_memory PRIVATE runtime/cuda)
    set_target_properties(test_cuda_pinned_memory PROPERTIES CUDA_ARCHITECTURES "120-real")
    add_test(NAME cuda_pinned_memory COMMAND test_cuda_pinned_memory)

    add_executable(test_cuda_async_pipeline tests/cuda/test_cuda_async_pipeline.cu)
    target_link_libraries(test_cuda_async_pipeline PRIVATE k3x_runtime)
    target_include_directories(test_cuda_async_pipeline PRIVATE runtime/cuda)
    set_target_properties(test_cuda_async_pipeline PROPERTIES CUDA_ARCHITECTURES "120-real")
    add_test(NAME cuda_async_pipeline COMMAND test_cuda_async_pipeline)

    add_executable(test_cuda_residency tests/cuda/test_cuda_residency.cu)
    target_link_libraries(test_cuda_residency PRIVATE k3x_runtime)
    target_include_directories(test_cuda_residency PRIVATE runtime/cuda)
    set_target_properties(test_cuda_residency PROPERTIES CUDA_ARCHITECTURES "120-real")
    add_test(NAME cuda_residency COMMAND test_cuda_residency)

    add_executable(test_cuda_situ tests/cuda/test_cuda_situ.cu)
    target_link_libraries(test_cuda_situ PRIVATE k3x_runtime)
    target_include_directories(test_cuda_situ PRIVATE runtime/cuda)
    set_target_properties(test_cuda_situ PROPERTIES CUDA_ARCHITECTURES "120-real")
    add_test(NAME cuda_situ COMMAND test_cuda_situ)

    add_executable(test_cuda_ffn tests/cuda/test_cuda_ffn.cu)
    target_link_libraries(test_cuda_ffn PRIVATE k3x_runtime)
    set_target_properties(test_cuda_ffn PROPERTIES CUDA_ARCHITECTURES "120-real")
    add_test(NAME cuda_ffn COMMAND test_cuda_ffn)

    add_executable(test_cuda_async_ffn tests/cuda/test_cuda_async_ffn.cu)
    target_link_libraries(test_cuda_async_ffn PRIVATE k3x_runtime)
    set_target_properties(test_cuda_async_ffn PROPERTIES CUDA_ARCHITECTURES "120-real")
    add_test(NAME cuda_async_ffn COMMAND test_cuda_async_ffn)
endif()

add_executable(test_reader tests/cpp/test_reader.cpp)
target_link_libraries(test_reader PRIVATE k3x_runtime)

add_executable(test_reader_concurrency tests/cpp/test_reader_concurrency.cpp)
target_link_libraries(test_reader_concurrency PRIVATE k3x_runtime)

add_executable(test_reader_contract tests/cpp/test_reader_contract.cpp)
target_link_libraries(test_reader_contract PRIVATE k3x_runtime)
add_test(NAME reader_contract COMMAND test_reader_contract)

add_executable(test_io_completion tests/cpp/test_io_completion.cpp)
target_include_directories(test_io_completion PRIVATE runtime/src)
target_link_libraries(test_io_completion PRIVATE k3x_runtime)
add_test(NAME io_completion COMMAND test_io_completion)

if(K3X_ENABLE_IO_URING)
    add_executable(test_io_ring_guard tests/cpp/test_io_ring_guard.cpp)
    target_compile_definitions(test_io_ring_guard PRIVATE K3X_ENABLE_IO_URING=1)
    target_include_directories(test_io_ring_guard PRIVATE
        runtime/src ${LIBURING_INCLUDE_DIR})
    target_link_libraries(test_io_ring_guard PRIVATE ${LIBURING_LIBRARY})
    add_test(NAME io_ring_guard COMMAND test_io_ring_guard)
endif()

add_executable(k3x_run runtime/src/main.cpp runtime/src/model.cpp)
target_link_libraries(k3x_run PRIVATE k3x_runtime)

add_executable(k3x_storage_bench runtime/src/storage_bench.cpp)
target_link_libraries(k3x_storage_bench PRIVATE k3x_runtime)

add_executable(test_model_session tests/cpp/test_model_session.cpp runtime/src/model.cpp)
target_link_libraries(test_model_session PRIVATE k3x_runtime)
