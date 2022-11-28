## Setting up OpenCL environment on dev machine for CPU
### Run these commands to compile and install OpenCL libraries on your system
```bash
mkdir opencl-config
cd opencl-config
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh
sudo apt-get install -y clinfo opencl-clhpp-headers build-essential ocl-icd-libopencl1 cmake git pkg-config libclang-dev clang make ninja-build ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev clinfo dialog apt-utils libxml2-dev libclang-cpp-dev
wget http://portablecl.org/downloads/pocl-1.6.tar.gz
tar -xvf pocl-1.6.tar.gz
cd pocl-1.6
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr -DENABLE_ICD=ON ..
sudo make CL_TARGET_OPENCL_VERSION=120 && sudo make install
```

### After that run clinfo to see detected hardware. The output should be similar to this:

```console
Number of platforms                               1
  Platform Name                                   Portable Computing Language
  Platform Vendor                                 The pocl project
  Platform Version                                OpenCL 1.2 pocl 1.6, None+Asserts, LLVM 9.0.1, RELOC, SLEEF, DISTRO, POCL_DEBUG
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_icd
  Platform Extensions function suffix             POCL

  Platform Name                                   Portable Computing Language
Number of devices                                 1
  Device Name                                     pthread-Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz
  Device Vendor                                   GenuineIntel
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 1.2 pocl HSTR: pthread-x86_64-pc-linux-gnu-skylake
  Driver Version                                  1.6
  Device OpenCL C Version                         OpenCL C 1.2 pocl
  Device Type                                     CPU
  Device Profile                                  FULL_PROFILE

```

## Setting up OpenCL for NVIDIA GPUs
### System configuration on azure:
* Instance:   NC6
* vCPU:       6
* RAM:        56GB
* GPU:        NVidia Tesla K80

#### Use the ubuntu-drivers devices command to fetch the name of your recommended driver
```console
  azureuser@myVm:~$ ubuntu-drivers devices
  
  == /sys/devices/LNXSYSTM:00/LNXSYBUS:00/PNP0A03:00/device:07/VMBUS:01/47505500-0001-0000-3130-444531303244/pci0001:00/0001:00:00.0 ==
  modalias : pci:v000010DEd0000102Dsv000010DEsd0000106Cbc03sc02i00
  vendor   : NVIDIA Corporation
  model    : GK210GL [Tesla K80]
  driver   : nvidia-driver-470-server - distro non-free
  driver   : nvidia-driver-418-server - distro non-free
  driver   : nvidia-driver-450-server - distro non-free
  driver   : nvidia-driver-470 - distro non-free recommended
  driver   : nvidia-driver-390 - distro non-free
  driver   : xserver-xorg-video-nouveau - distro free builtin
```

Above, note that the recommended driver is __nvidia-driver-470__

#### Install the recommended driver along-with CUDA. At least **5GB** of free space is required!
```console
sudo apt install nvidia-driver-470 nvidia-cuda-toolkit
```
#### After all the above three packages are installed, restart your system and verify your OpenCL configuration with clinfo
```console
azureuser@myVm:~/nvidia-opencl$ clinfo
Number of platforms                               1
  Platform Name                                   NVIDIA CUDA
  Platform Vendor                                 NVIDIA Corporation
  Platform Version                                OpenCL 3.0 CUDA 11.4.264
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_icd cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts cl_nv_create_buffer cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_nv_kernel_attribute cl_khr_device_uuid cl_khr_pci_bus_info
  Platform Host timer resolution                  0ns
  Platform Extensions function suffix             NV

  Platform Name                                   NVIDIA CUDA
Number of devices                                 1
  Device Name                                     Tesla K80
  Device Vendor                                   NVIDIA Corporation
  Device Vendor ID                                0x10de
  Device Version                                  OpenCL 3.0 CUDA
  Driver Version                                  470.141.03
  Device OpenCL C Version                         OpenCL C 1.2
  Device Type                                     GPU
  Device Topology (NV)                            PCI-E, 00:00.0
  Device Profile                                  FULL_PROFILE
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Max compute units                               13
  Max clock frequency                             823MHz
  Compute Capability (NV)                         3.7
  Device Partition                                (core)
    Max number of sub-devices                     1
    Supported partition types                     None
    Supported affinity domains                    (n/a)
  Max work item dimensions                        3
  Max work item sizes                             1024x1024x64
  Max work group size                             1024
  Preferred work group size multiple              32
  Warp size (NV)                                  32
  Max sub-groups per work group                   0
  Preferred / native vector sizes
    char                                                 1 / 1
    short                                                1 / 1
    int                                                  1 / 1
    long                                                 1 / 1
    half                                                 0 / 0        (n/a)
    float                                                1 / 1
    double                                               1 / 1        (cl_khr_fp64)
  Half-precision Floating-point support           (n/a)
  Single-precision Floating-point support         (core)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  Yes
  Double-precision Floating-point support         (cl_khr_fp64)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
  Address bits                                    64, Little-Endian
  Global memory size                              11997020160 (11.17GiB)
  Error Correction support                        Yes
  Max memory allocation                           2999255040 (2.793GiB)
  Unified memory for Host and Device              No
  Integrated memory (NV)                          No
  Shared Virtual Memory (SVM) capabilities        (core)
    Coarse-grained buffer sharing                 Yes
    Fine-grained buffer sharing                   No
    Fine-grained system sharing                   No
    Atomics                                       No
  Minimum alignment for any data type             128 bytes
  Alignment of base address                       4096 bits (512 bytes)
  Preferred alignment for atomics
    SVM                                           0 bytes
    Global                                        0 bytes
    Local                                         0 bytes
  Max size for global variable                    0
  Preferred total size of global vars             0
  Global Memory cache type                        Read/Write
  Global Memory cache size                        212992 (208KiB)
  Global Memory cache line size                   128 bytes
  Image support                                   Yes
    Max number of samplers per kernel             32
    Max size for 1D images from buffer            134217728 pixels
    Max 1D or 2D image array size                 2048 images
    Max 2D image size                             16384x16384 pixels
    Max 3D image size                             4096x4096x4096 pixels
    Max number of read image args                 256
    Max number of write image args                16
    Max number of read/write image args           0
  Max number of pipe args                         0
  Max active pipe reservations                    0
  Max pipe packet size                            0
  Local memory type                               Local
  Local memory size                               49152 (48KiB)
  Registers per block (NV)                        65536
  Max number of constant args                     9
  Max constant buffer size                        65536 (64KiB)
  Max size of kernel argument                     4352 (4.25KiB)
  Queue properties (on host)
    Out-of-order execution                        Yes
    Profiling                                     Yes
  Queue properties (on device)
    Out-of-order execution                        No
    Profiling                                     No
    Preferred size                                0
    Max size                                      0
  Max queues on device                            0
  Max events on device                            0
  Prefer user sync for interop                    No
  Profiling timer resolution                      1000ns
  Execution capabilities
    Run OpenCL kernels                            Yes
    Run native kernels                            No
    Sub-group independent forward progress        No
    Kernel execution timeout (NV)                 No
  Concurrent copy and kernel execution (NV)       Yes
    Number of async copy engines                  2
    IL version                                    (n/a)
  printf() buffer size                            1048576 (1024KiB)
  Built-in kernels                                (n/a)
  Device Extensions                               cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_icd cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts cl_nv_create_buffer cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_nv_kernel_attribute cl_khr_device_uuid cl_khr_pci_bus_info

NULL platform behavior
  clGetPlatformInfo(NULL, CL_PLATFORM_NAME, ...)  NVIDIA CUDA
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, ...)   Success [NV]
  clCreateContext(NULL, ...) [default]            Success [NV]
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_DEFAULT)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CUSTOM)  Invalid device type for platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL)  No platform

ICD loader properties
  ICD loader Name                                 OpenCL ICD Loader
  ICD loader Vendor                               OCL Icd free software
  ICD loader Version                              2.2.11
  ICD loader Profile                              OpenCL 2.1
        NOTE:   your OpenCL library only supports OpenCL 2.1,
                but some installed platforms support OpenCL 3.0.
                Programs using 3.0 features may crash
                or behave unexpectedly

```

## OpenCL on Docker for NVIDIA GPUs

### Install the NVIDIA Container Runtime
#### First add the repository details then install nvidia-container-runtime
```console
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
  
sudo apt update
sudo apt install nvidia-container-runtime
```

### Building the Dockerfile

```console
sudo docker build -t nvidia-opencl .
```

### Launch the OpenCL Container
```console
docker run --rm -it --gpus nvidia-opencl
```