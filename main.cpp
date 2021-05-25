#include "scan.hpp"
#include "const.hpp"
#include <string>

int main(int argc, char* argv[]) {
    DeviceType dt;
    if (argc < 3) {
        std::cout << "Please specify device type and algorithm with: " << argv[0] << " gpu|cpu|igpu " << "scan|const\n";
        exit(1);
    }
    if (std::string(argv[1]) == "gpu")
        dt = DeviceType::GPU;
    else if (std::string(argv[1]) == "cpu")
        dt = DeviceType::CPU;
    else if (std::string(argv[1]) == "igpu")
        dt = DeviceType::iGPU;
    else {
        std::cout << "Please specify device type and algorithm with: " << argv[0] << " gpu|cpu|igpu " << "scan|const\n";
        exit(1);
    }

    if (std::string(argv[2]) == "scan") {
        if (dt == DeviceType::GPU)
            scan_cuda(dt);
        else
            scan(dt);
        }
    else if (std::string(argv[2]) == "const") {
        if (dt == DeviceType::GPU)
            constant_cuda(dt);
        else
            constant(dt);
    } else {
        std::cout << "Please specify device type and algorithm with: " << argv[0] << " gpu|cpu|igpu " << "scan|const\n";
        exit(1);
    }
}
