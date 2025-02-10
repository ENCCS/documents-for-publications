// Initialize OpenCL

cl::Device device = cl::Device::getDefault();
cl::Context context(device);
cl::CommandQueue queue(context, device);
