cmake_minimum_required(VERSION 3.5.0)
project(Deep_Library_CPP VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(Deep_Library_CPP src/main.cpp)
target_include_directories(Deep_Library_CPP PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(Deep_Library_CPP PUBLIC cxx_std_23)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
target_compile_options(Deep_Library_CPP PUBLIC -O3 -mavx2 -mfma -msse4.1 -msse4.2)
endif()

add_executable(benchmark benchmarks/benchmark.cpp)
target_include_directories(benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(benchmark PUBLIC cxx_std_23)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
target_compile_options(benchmark PUBLIC -O3 -mavx2 -mfma -msse4.1 -msse4.2)
endif()