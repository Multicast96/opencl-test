cmake_minimum_required(VERSION 3.18)
project(opencl_example)

set(CMAKE_CXX_STANDARD 20)
configure_file(kernel.cl kernel.cl COPYONLY)

add_executable(opencl_example main.cpp)

find_package(OpenCL REQUIRED)
target_link_libraries(opencl_example OpenCL::OpenCL)
