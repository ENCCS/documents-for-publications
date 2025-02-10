// Initialize OpenCL

// Error code returned by API calls
cl_int err;
cl_platform_id platform;
err = clGetPlatformIDs(1, &platform, NULL);

// Checking error codes is skipped later for brevity
assert(err == CL_SUCCESS); 
cl_device_id device;
err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
