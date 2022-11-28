#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <iostream>
#include <CL/opencl.hpp>
#include <fstream>
#include <chrono>

void printSystemInfo(const cl::Device &device);

void computeInParallel(std::vector<int> &, std::vector<int> &, cl::Context &, cl::Program &, cl::Device &);

void computeInSequence(std::vector<int> &, const std::vector<int> &);

void checkResult(const std::vector<int> &result);

const int VECTOR_SIZE = 100'000'000;
const std::string KERNEL_PROGRAM_FILE = "array_addition.cl";

int main() {
    // prepare input data
    std::vector<int> a(VECTOR_SIZE), b(VECTOR_SIZE), result(VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i;
        b[i] = VECTOR_SIZE - i;
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

    sources.push_back({src.c_str(), src.length()});
    context = cl::Context(device);
    program = cl::Program(context, sources);

    auto err = program.build();
    if (err != CL_BUILD_SUCCESS) {
        std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) << std::endl;
        exit(1);
    } else {
        std::cout << "Kernel program " << KERNEL_PROGRAM_FILE << " build success\n";
    }

    auto &vc = a;
    cl::Buffer aBuf(context, CL_MEM_USE_HOST_PTR, sizeof(int) * VECTOR_SIZE, vc.data());
    computeInSequence(a, b);
    computeInParallel(a, b, context, program, device);
}

void computeInSequence(std::vector<int> &a, const std::vector<int> &b) {
    std::vector<int> result(VECTOR_SIZE);
    std::cout << "Compute addition of " << VECTOR_SIZE << " elements in sequence started\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    for (auto i = 0; i < VECTOR_SIZE; i++) {
        result[i] = a[i] + b[i];
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;
    checkResult(result);
    std::cout << "Task finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms\n";
}

void computeInParallel(std::vector<int> &a, std::vector<int> &b, cl::Context &context, cl::Program &program,
                       cl::Device &device) {
    std::vector<int> result(VECTOR_SIZE);
    std::cout << "Compute addition of " << VECTOR_SIZE << " elements in parallel started\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    // Parallely performs the operation c = a + b.

    // Create buffers and allocate memory on the device.
    cl::Buffer aBuf(context, CL_MEM_USE_HOST_PTR, sizeof(int) * VECTOR_SIZE, a.data());
    cl::Buffer bBuf(context, CL_MEM_USE_HOST_PTR, sizeof(int) * VECTOR_SIZE, b.data());
    cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY, sizeof(int) * VECTOR_SIZE);

    // create the kernel functor
    int32_t error = 0;
    cl::Kernel kernel(program, "vadd", &error);

    if (error != 0) {
        if (error == CL_INVALID_KERNEL_NAME) {
            std::cerr << "Invalid kernel name" << std::endl;
            std::exit(1);
        }
    }
    kernel.setArg(0, aBuf);
    kernel.setArg(1, bBuf);
    kernel.setArg(2, cBuf);

    // Run the kernel function and collect its result.
    cl::CommandQueue queue(context, device);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(VECTOR_SIZE), cl::NDRange(8));
    queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, sizeof(int) * VECTOR_SIZE, result.data());
    queue.finish();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;
    std::cout << "Task finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms\n";
}

void checkResult(const std::vector<int> &result) {
    if (result.size() != VECTOR_SIZE) {
        std::cerr << "Vector size should equal " << VECTOR_SIZE << " but it's " << result.size() << std::endl;
        std::exit(1);
    }

    for (const auto item: result) {
        if (item != VECTOR_SIZE) {
            std::cerr << "All vector items should equal " << VECTOR_SIZE << " but one is " << item << std::endl;
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

