cl_int err;
cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);

err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
cl_kernel kernel_dot = clCreateKernel(program, "vector_add", &err);