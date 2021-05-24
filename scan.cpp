#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <CL/sycl.hpp>

#include "scan.hpp"

class CUDASelector : public cl::sycl::device_selector {
  public:
    int operator()(const cl::sycl::device &Device) const override {
      using namespace cl::sycl::info;
      const std::string DriverVersion = Device.get_info<device::driver_version>();

      if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos)) {
        return 1;
      };
      return -1;
    }
};

std::unique_ptr<cl::sycl::device_selector>
get_device_selector(DeviceType dt) {
  using namespace cl::sycl;
  switch (dt) {
  case DeviceType::CPU:
    return std::make_unique<cpu_selector>();
  case DeviceType::GPU:
    return std::make_unique<CUDASelector>();
  case DeviceType::iGPU:
    return std::make_unique<gpu_selector>();

  default:
    throw std::logic_error("Unsupported device type.");
  }
}


template <typename T> using Func = std::function<bool(T)>;
template <typename T>
std::vector<T> expected_out(const std::vector<T> &v, Func<T> f) {
  std::vector<int> out;
  std::copy_if(v.begin(), v.end(), std::back_inserter(out), f);
  return out;
}

void scan(DeviceType dt) {
  const size_t buf_size = 8;
  const std::vector<int> host_src = {1, 2, 3, 4, 5, 6, 7, 8};

  std::vector<int> expected =
      expected_out<int>(host_src, [](int x) { return x < 5; });

  auto sel = get_device_selector(dt);
  sycl::queue q{*sel.get()};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  auto dev_policy = oneapi::dpl::execution::device_policy{*sel};

  sycl::buffer<int> src_buf(host_src.data(), sycl::range<1>{buf_size});
  sycl::buffer<int> out_buf{sycl::range<1>{buf_size}};

  auto host_start = std::chrono::steady_clock::now();

  auto end_it = std::copy_if(
        dev_policy, oneapi::dpl::begin(src_buf), oneapi::dpl::end(src_buf),
        oneapi::dpl::begin(out_buf), [](auto &x) { return x < 5; });

  auto host_end = std::chrono::steady_clock::now();
  auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();
  {
    sycl::host_accessor res(out_buf, sycl::read_only);
    std::cout << "Input:    ";
    dump_collection(host_src);
    std::cout << "Output:    ";
    for (int i = 0; i < expected.size(); ++i) {
      std::cout << res[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Expected: ";
    dump_collection(expected);
  }
}

