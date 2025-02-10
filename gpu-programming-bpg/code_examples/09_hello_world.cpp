//
// CUDA
//
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

int main(void)
{
    int count, device;

    cudaGetDeviceCount(&count);
    cudaGetDevice(&device);

    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);
    return 0;
}





//
// HIP
//
#include <hip/hip_runtime.h>
#include <stdio.h>

int main(void)
{
    int count, device;

    hipGetDeviceCount(&count);
    hipGetDevice(&device);

    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);
    return 0;
}





//
// Kokkos
//
#include <Kokkos_Core.hpp>
#include <iostream>

int main()
{
    Kokkos::initialize();

    int count = Kokkos::Cuda().concurrency();
    int device = Kokkos::Cuda().impl_internal_space_instance()->impl_internal_space_id();

    std::cout << "Hello! I'm GPU " << device << " out of "
              << count << " GPUs in total." << std::endl;

    Kokkos::finalize();
    return 0;
}





//
// OpenCL
//
#include <CL/opencl.h>
#include <stdio.h>

int main(void)
{
    cl_uint count;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &count);

    char deviceName[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);

    printf("Hello! I'm GPU %s out of %d GPUs in total.\n", deviceName, count);

    return 0;
}





//
// SYCL
//
#include <iostream>
#include <sycl/sycl.hpp>

int main()
{
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    auto count = gpu_devices.size();
    std::cout << "Hello! I'm using a SYCL device by "
              << gpu_devices[0].get_info<sycl::info::device::vendor>()
              << ">, the first of " << count << " devices." << std::endl;
    return 0;
}

