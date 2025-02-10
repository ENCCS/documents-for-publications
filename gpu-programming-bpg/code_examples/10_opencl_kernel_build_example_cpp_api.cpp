cl::Program program(context, kernel_source);
program.build({device});
cl::Kernel kernel_dot(program, "dot");
