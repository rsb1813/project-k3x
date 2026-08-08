cmake_minimum_required(VERSION 3.25)
project(k3x LANGUAGES CXX)

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

add_executable(test_reader tests/cpp/test_reader.cpp)
target_link_libraries(test_reader PRIVATE k3x_runtime)

add_executable(k3x_run runtime/src/main.cpp runtime/src/model.cpp)
target_link_libraries(k3x_run PRIVATE k3x_runtime)
