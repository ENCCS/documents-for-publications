// Create a buffer of n integers
auto buf = sycl::buffer<int>(sycl::range<1>(n));

// Submit a kernel into a queue; cgh is a helper object
q.submit([&](sycl::handler &cgh)
{
    // Create write-only accessor for buf
    auto acc = buf.get_access<sycl::access_mode::write>(cgh);
  
    // Define a kernel: n threads execute the following lambda
    cgh.parallel_for<class KernelName>(sycl::range<1>{n}, [=](sycl::id<1> i)
    {
        // The data is written to the buffer via acc
        acc[i] = /*...*/
    });
});

// If we now submit another kernel with accessor to buf, it will not start running until the kernel above is done
