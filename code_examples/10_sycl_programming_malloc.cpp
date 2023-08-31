// Create a shared (migratable) allocation of n integers
// Unlike with buffers, we need to specify a queue (or, explicitly, a device and a context)
int* v = sycl::malloc_shared<int>(n, q);

// Submit a kernel into a queue; cgh is a helper object
q.submit([&](sycl::handler &cgh)
{
	// Define a kernel: n threads execute the following lambda
	cgh.parallel_for<class KernelName>(sycl::range<1>{n}, [=](sycl::id<1> i)
	{
		// The data is directly written to v
		v[i] = /*...*/
	});
});

// If we want to access v, we have to ensure that the kernel has finished
q.wait();

// After we're done, the memory must be deallocated
sycl::free(v, q);
