cmake_minimum_required(VERSION 3.16)
project(autodiff LANGUAGES CXX)

add_library(autodiff STATIC src/cpp/tensor.cpp)
target_include_directories(autodiff PUBLIC include)
target_compile_features(autodiff PUBLIC cxx_std_17)

option(AUTODIFF_BUILD_TESTS "Build the C++ unit tests" ON)
option(AUTODIFF_BUILD_BENCHMARKS "Build the C++ benchmark" ON)

if(AUTODIFF_BUILD_TESTS)
  enable_testing()
  add_executable(autodiff_cpp_tests tests/test_tensor_cpp.cpp)
  target_link_libraries(autodiff_cpp_tests PRIVATE autodiff)
  add_test(NAME autodiff_cpp_tests COMMAND autodiff_cpp_tests)
endif()

if(AUTODIFF_BUILD_BENCHMARKS)
  add_executable(autodiff_cpp_benchmark benchmarks/cpp_benchmark.cpp)
  target_link_libraries(autodiff_cpp_benchmark PRIVATE autodiff)
endif()
