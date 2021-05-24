#include "scan.hpp"
#include <string>

int main(int argc, char* argv[]) {
    DeviceType dt;
    if (argc < 2) {
        std::cout << "Please specify device type with: " << argv[0] << " gpu|cpu|igpu\n";
        exit(1);
    }
    if (std::string(argv[1]) == "gpu")
        dt = DeviceType::GPU;
    else if (std::string(argv[1]) == "cpu")
        dt = DeviceType::CPU;
    else if (std::string(argv[1]) == "igpu")
        dt = DeviceType::iGPU;
    else {
        std::cout << "Please specify device type with: " << argv[0] << " gpu|cpu|igpu\n";
        exit(1);
    }

    scan(dt);
}
