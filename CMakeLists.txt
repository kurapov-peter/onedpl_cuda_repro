cmake_minimum_required(VERSION 3.18)
project(cmake_sycl)

find_package(oneDPL REQUIRED)

add_library(scan SHARED scan.cpp)
target_link_libraries(scan PRIVATE sycl oneDPL)
target_compile_options(scan PRIVATE -fsycl)
target_link_options(scan PRIVATE -fsycl)

add_library(scan_cuda SHARED scan.cpp)
target_link_libraries(scan_cuda PRIVATE sycl oneDPL)
target_compile_options(scan_cuda PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
target_link_options(scan_cuda PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)

add_executable(main main.cpp)
target_link_libraries(main scan scan_cuda)
target_link_options(main PRIVATE -fsycl)

#add_executable(main_cuda main.cpp)
#target_link_libraries(main_cuda scan_cuda)
