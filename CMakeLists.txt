cmake_minimum_required(VERSION 3.18)
project(cmake_sycl)

find_package(oneDPL REQUIRED)

add_library(scan scan.cpp)
target_link_libraries(scan PRIVATE sycl oneDPL)
target_compile_options(scan PRIVATE -fsycl -fsycl-unnamed-lambda)
target_link_options(scan PRIVATE -fsycl -fsycl-unnamed-lambda)

# Enable to get Error: Device name missing. llvm-foreach: clang-13: error: gen compiler command failed with exit code 1
#target_compile_options(scan PRIVATE -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice,spir64_x86_64-unknown-unknown-sycldevice)
#target_link_options(scan PRIVATE -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice,spir64_x86_64-unknown-unknown-sycldevice)

add_library(scan_cuda scan_cuda.cpp)
target_link_libraries(scan_cuda PRIVATE sycl oneDPL)
target_compile_options(scan_cuda PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda)
target_link_options(scan_cuda PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda)

add_executable(main main.cpp)
# Separate binaries work fine
#target_link_libraries(main scan)
target_link_libraries(main scan_cuda scan)
#target_link_options(main PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)

# Separate binaries work fine
#add_executable(main_cuda main.cpp)
#target_link_libraries(main_cuda scan_cuda)
