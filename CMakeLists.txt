cmake_minimum_required(VERSION 3.18)
project(cmake_sycl)

find_package(oneDPL REQUIRED)

add_executable(scan scan.cpp)
target_link_libraries(scan sycl oneDPL)
target_compile_options(scan PRIVATE -fsycl)
target_link_options(scan PRIVATE -fsycl)

add_executable(scan_cuda scan.cpp)
target_link_libraries(scan_cuda sycl oneDPL)
target_compile_options(scan_cuda PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
target_link_options(scan_cuda PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
