#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "extra/cl.hpp"

static std::string dev_type_name(unsigned dev_type) {
  std::string ret;
  if (dev_type & CL_DEVICE_TYPE_CPU)
    ret += "cpu ";
  if (dev_type & CL_DEVICE_TYPE_GPU)
    ret += "gpu ";
  if (dev_type & CL_DEVICE_TYPE_ACCELERATOR)
    ret += "accel ";
  if (dev_type & CL_DEVICE_TYPE_CUSTOM)
    ret += "custom ";
  return ret;
}

// Eumerate and select device: "cpu", "gpu", or ""
static cl::Device select_device(const std::string &where) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty())
    throw cl::Error(-1, "No platforms found");

  cl::Device selected_dev;

  for (auto &plat : platforms) {
    std::cout << "Platform \"" << plat.getInfo<CL_PLATFORM_NAME>()
              << "\". Devices:\n";

    std::vector<cl::Device> devices;
    plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.size() == 0) {
      std::cout << "  No devices found.\n";
      continue;
    }

    for (auto &dev : devices) {
      auto dev_type = dev.getInfo<CL_DEVICE_TYPE>();
      std::cout << " - [" << dev_type_name(dev_type) << "] "
                << dev.getInfo<CL_DEVICE_VENDOR>() << ": "
                << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
      std::cout << "   (Max compute units: "
                << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
                << ", max work group size: "
                << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << ")\n";

      if (selected_dev())
        continue;

      if (where == "cpu" && dev_type == CL_DEVICE_TYPE_CPU) {
        selected_dev = dev;
      } else if (where == "gpu" && dev_type == CL_DEVICE_TYPE_GPU) {
        selected_dev = dev;
      }
    }
    std::cout << std::endl;
  }

  if (where.empty())
    selected_dev = cl::Device::getDefault();

  if (!selected_dev())
    throw cl::Error(-1, "Device not found");

  return selected_dev;
}
