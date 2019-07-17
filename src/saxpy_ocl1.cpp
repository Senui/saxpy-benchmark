#include "saxpy.h"
#include <iostream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "extra/cl.hpp"

// For some docs:
// - http://github.khronos.org/OpenCL-CLHPP/
// -
// http://simpleopencl.blogspot.co.id/2013/06/tutorial-simple-start-with-opencl-and-c.html


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

int main(int argc, const char *argv[]) {
  try {
    // Search for available devices and select the first of the available type
    // (either CPU or GPU, depending on the command line argument)
    cl::Device default_device = select_device(argc > 1 ? argv[1] : "");
    std::cout << "Using " << default_device.getInfo<CL_DEVICE_VENDOR>() << " "
              << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;

    // OpenCL boilerplate code
    cl::Context context(default_device);
    cl::Program::Sources sources;

    // The OpenCL kernel code (i.e. the code that will be executed on the GPU)
    std::string kernel_code = "__kernel void saxpy(float alpha,"
                              "                    __global const float* X,"
                              "                    __global float* Y)"
                              "{"
                              "    int i = get_global_id(0); "
                              "    Y[i] += alpha * X[i];"
                              "}";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    // Compile the GPU code
    cl::Program program(context, sources);
    program.build({default_device});

    // Allocate GPU memory buffers
    cl::Buffer dev_x(context, CL_MEM_READ_ONLY, N * sizeof(float));
    cl::Buffer dev_y(context, CL_MEM_READ_WRITE, N * sizeof(float));

    // Initialize arrays on the host (CPU)
    std::vector<float> host_x(N), host_y(N);
    for (size_t i = 0; i < N; ++i) {
      host_x[i] = XVAL;
      host_y[i] = YVAL;
    }

    // Write the initialized CPU arrays to the allocated GPU buffers
    cl::CommandQueue queue(context, default_device);
    queue.enqueueWriteBuffer(dev_x, CL_TRUE, 0, N * sizeof(float), host_x.data());
    queue.enqueueWriteBuffer(dev_y, CL_TRUE, 0, N * sizeof(float), host_y.data());

    // Create a callable object that represents the GPU kernel 
    cl::Kernel kernel(program, "saxpy");

    // The buffers we created can now be used as function arguments for the GPU
    // kernel program
    kernel.setArg(0, (float)AVAL);
    kernel.setArg(1, dev_x);
    kernel.setArg(2, dev_y);

    // A wall clock timer (starts the timer upon construction)
    saxpy_timer timer;

    // Invoke the kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N));
    
    // Wait for the GPU to finish with execution
    queue.finish();

    // Measure how much time elapsed since the timer started
    double elapsed = timer.elapsed_msec();
    std::cout << "Elapsed: " << elapsed << " ms\n";

    // Read out the result (Y)
    queue.enqueueReadBuffer(dev_y, CL_TRUE, 0, N * sizeof(float), host_y.data());

    // Verify the correctness
    saxpy_verify(host_y);
  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << "(code: " << err.err() << ")\n";
  }
}
