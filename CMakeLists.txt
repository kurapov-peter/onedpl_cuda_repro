cmake_minimum_required(VERSION 3.18)
project(cmake_sycl)

find_package(oneDPL REQUIRED)

add_library(scan SHARED scan.cpp)
target_link_libraries(scan PRIVATE sycl oneDPL)
target_compile_options(scan PRIVATE -fsycl)
target_link_options(scan PRIVATE -fsycl)

# Enable to get Error: Device name missing. llvm-foreach: clang-13: error: gen compiler command failed with exit code 1
#target_compile_options(scan PRIVATE -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice,spir64_x86_64-unknown-unknown-sycldevice)
#target_link_options(scan PRIVATE -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice,spir64_x86_64-unknown-unknown-sycldevice)

add_library(scan_cuda SHARED scan.cpp)
target_link_libraries(scan_cuda PRIVATE sycl oneDPL)
target_compile_options(scan_cuda PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
target_link_options(scan_cuda PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)

add_executable(main main.cpp)
# Separate binaries work fine
#target_link_libraries(main scan)
target_link_libraries(main scan scan_cuda)
target_link_options(main PRIVATE -fsycl)

# Separate binaries work fine
#add_executable(main_cuda main.cpp)
#target_link_libraries(main_cuda scan_cuda)
