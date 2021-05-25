#include <oneapi/dpl/execution>

#include <CL/sycl.hpp>

#include "const.hpp"

namespace {
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

}

void constant(DeviceType dt) {
  const size_t buf_size = 8;
  const std::vector<int> host_src = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<int> expected = {2, 3, 4, 5, 6, 7, 8, 9};

  auto sel = get_device_selector(dt);
  using namespace sycl;
  auto rng = range<1>{buf_size};
  sycl::queue q{*sel.get()};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  auto dev_policy = oneapi::dpl::execution::device_policy{*sel};

  sycl::buffer<int> src_buf(host_src.data(), sycl::range<1>{buf_size});

  q.submit([&](handler &h) {
    auto out = src_buf.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for<class hello>(rng, [=](auto &idx) { out[idx] = out[idx] + 1; });
  });

  {
    sycl::host_accessor res(src_buf, sycl::read_only);
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
