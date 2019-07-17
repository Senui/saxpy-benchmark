#include <iostream>
#include <vector>

#include "saxpy.h"
#include "saxpy_ocl1_helper.h"

#define __CL_ENABLE_EXCEPTIONS
#include "extra/cl.hpp"

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
    cl::CommandQueue queue(context, default_device, CL_QUEUE_PROFILING_ENABLE);
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

    // An OpenCL Event: useful for getting profiling information from OpenCL calls
    cl::Event evt;

    // Invoke the kernel. This is an "non-blocking" call. This means that the
    // CPU can go on with doing whatever it likes, because this workload is
    // running on the GPU!
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NullRange, NULL, &evt);

    // Wait for the GPU to finish execution
    evt.wait();

    // Get the profiling data from the Event object
    cl_ulong time_start, time_end;
    evt.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    evt.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Print out the kernel execution time
    double kernel_time = (time_end - time_start) / 1000000.0;
    printf("GPU execution time = %0.3f ms \n", kernel_time);
    
    // This forces the CPU to wait for the GPU to finish with execution
    queue.finish();

    // Read out the result (Y)
    queue.enqueueReadBuffer(dev_y, CL_TRUE, 0, N * sizeof(float), host_y.data());

    // Verify the correctness
    saxpy_verify(host_y);
  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << "(code: " << err.err() << ")\n";
  }
}
