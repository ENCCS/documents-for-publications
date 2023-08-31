static const std::string kernel_source = R"(
	__kernel void dot(__global int *a) {
		int i = get_global_id(0);
		a[i] = i;
	}
)";
