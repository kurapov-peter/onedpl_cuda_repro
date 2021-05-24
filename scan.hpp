#include <iostream>

enum DeviceType {CPU, GPU, iGPU};

template <class Collection>
void dump_collection(const Collection &c, std::ostream &os = std::cout) {
  bool first = true;
  for (const auto &e : c) {
    if (!first) {
      os << " ";
    }
    os << e;
    first = false;
  }
  os << "\n";
}

void scan(DeviceType dt);
void scan_cuda(DeviceType dt);
