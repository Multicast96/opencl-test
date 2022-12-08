#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <iostream>
#include <CL/opencl.hpp>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <iomanip>
#include <numbers>

void printSystemInfo(const cl::Device &device);

void computeInParallel(std::vector<float> &, std::vector<float> &, cl::Context &, cl::Program &, cl::Device &);

void computeInSequence(std::vector<float> &, const std::vector<float> &);

void checkResult(const std::vector<float> &result, const std::vector<float> &, const std::vector<float> &);

const int VECTOR_SIZE = 1'572'864;
const std::string KERNEL_PROGRAM_FILE = "kernel.cl";
const float SCALAR = std::numbers::pi;


bool areSame(float a, float b) {
    std::cout << std::fixed << std::showpoint << std::setprecision(std::numeric_limits<float>::digits);
    return std::fabs(a - b) < 1e-2;
}

float kernel(float a, float xi, float yi) {
    return a * xi + yi * xi;
}

int main() {
    // prepare input data
    srand(static_cast <unsigned> (time(0)));
    const int MAX_VALUE = 100;
    std::vector<float> a(VECTOR_SIZE), b(VECTOR_SIZE), result(VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / MAX_VALUE));
        b[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / MAX_VALUE));
    }

    // Search for all the OpenCL platforms available and check if there are any.
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No platforms found!" << std::endl;
        exit(1);
    } else {
        std::cout << "Platforms found: " << platforms.size() << std::endl;
    }

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    // Can be modified to devices CL_DEVICE_TYPE_GPU/CL_DEVICE_TYPE_CPU to select specified device
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    if (devices.empty()) {
        std::cerr << "No devices found!" << std::endl;
        exit(1);
    } else {
        std::cout << "Devices found: " << devices.size() << std::endl;
    }

    std::for_each(devices.begin(), devices.end(), printSystemInfo);


    // Read OpenCL kernel file as a string.
    std::ifstream kernel_file(KERNEL_PROGRAM_FILE);
    std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

    if (src.empty()) {
        std::cerr << "Kernel source file is empty!\n";
        std::exit(1);
    }

    // Compile kernel program which will run on the device.
    cl::Program::Sources sources;
    cl::Program program;    // The program that will run on the device.
    cl::Context context;    // The context which holds the device.
    cl::Device device = devices.front();      // The device where the kernel will run.

    sources.emplace_back(src.c_str(), src.length());
    context = cl::Context(device);
    program = cl::Program(context, sources);

    auto err = program.build();
    if (err != CL_BUILD_SUCCESS) {
        std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) << std::endl;
        exit(1);
    } else {
        std::cout << "Kernel program " << KERNEL_PROGRAM_FILE << " build success\n";
    }

    computeInSequence(a, b);
    computeInParallel(a, b, context, program, device);
}

void computeInSequence(std::vector<float> &a, const std::vector<float> &b) {
    std::vector<float> result(VECTOR_SIZE);
    std::cout << "Compute addition of " << VECTOR_SIZE << " elements in sequence started\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    for (auto i = 0; i < VECTOR_SIZE; i++) {
        result[i] = kernel(SCALAR, a[i], b[i]);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;
    checkResult(result, a, b);
    std::cout << "Task finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms\n";
}

void computeInParallel(std::vector<float> &a, std::vector<float> &b, cl::Context &context, cl::Program &program,
                       cl::Device &device) {
    std::vector<float> result(VECTOR_SIZE);
    // Parallely performs the operations.

    // Create buffers and allocate memory on the device.
    cl::Buffer aBuf(context, CL_MEM_USE_HOST_PTR, sizeof(float) * VECTOR_SIZE, a.data());
    cl::Buffer bBuf(context, CL_MEM_USE_HOST_PTR, sizeof(float) * VECTOR_SIZE, b.data());
    cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY, sizeof(float) * VECTOR_SIZE);

    // create the kernel functor
    int32_t error = 0;
    cl::Kernel kernel(program, "vadd", &error);

    if (error != 0) {
        if (error == CL_INVALID_KERNEL_NAME) {
            std::cerr << "Invalid kernel name" << std::endl;
            std::exit(1);
        }
    }
    kernel.setArg(0, SCALAR);
    kernel.setArg(1, aBuf);
    kernel.setArg(2, bBuf);
    kernel.setArg(3, cBuf);

    cl::Event computeEvent;

    // Run the kernel function and collect its result.
    cl::CommandQueue queue(context, device);

    std::cout << "Compute addition of " << VECTOR_SIZE << " elements in parallel started\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(VECTOR_SIZE), cl::NDRange(12), nullptr,
                               &computeEvent);
    computeEvent.wait();
    auto end_time = std::chrono::high_resolution_clock::now();

    queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, result.data());
    queue.finish();

    auto time = end_time - start_time;
    checkResult(result, a, b);
    std::cout << "Task finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms\n";
}

void checkResult(const std::vector<float> &result, const std::vector<float> &a, const std::vector<float> &b) {
    if (result.size() != VECTOR_SIZE) {
        std::cerr << "Vector size should equal " << VECTOR_SIZE << " but it's " << result.size() << std::endl;
        std::exit(1);
    }

    for (int i = 0; i < VECTOR_SIZE; i++) {
        const float singleResult = kernel(SCALAR, a[i], b[i]);
        if (!areSame(result[i], singleResult)) {
            std::cout << "Vector item #" << i << " should equal " << singleResult << " but is " << result[i]
                      << std::endl;
            std::exit(1);
        }
    }
}

void printSystemInfo(const cl::Device &device) {
    auto name = device.getInfo<CL_DEVICE_NAME>();
    auto vendor = device.getInfo<CL_DEVICE_VENDOR>();
    auto version = device.getInfo<CL_DEVICE_VERSION>();
    auto workItems = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    auto workGroups = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    auto computeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    auto globalMemory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    auto localMemory = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

    std::cout << "OpenCL Device Info:"
              << "\nName: " << name
              << "\nVendor: " << vendor
              << "\nVersion: " << version
              << "\nMax size of work-items: (" << workItems[0] << "," << workItems[1] << "," << workItems[2] << ")"
              << "\nMax size of work-groups: " << workGroups
              << "\nNumber of compute units: " << computeUnits
              << "\nGlobal memory size (bytes): " << globalMemory
              << "\nLocal memory size per compute unit (bytes): " << localMemory / computeUnits
              << std::endl;
}

