// SYCL has built-in sycl::reduction primitive, the use of which is demonstrated in
// the "Portable kernel models" chapter. Here is how the reduction can be implemented manually:

auto reductionKernel(sycl::handler &cgh, double *x, double *sum, int N)
{
    sycl::local_accessor<double, 1> shtmp{{2*tpb}, cgh};
    return [=](sycl::nd_item<1> item)
    {
        int ibl = item.get_group(0);
        int ind = item.get_global_id(0);
        int tid = item.get_local_id(0);
        shtmp[tid] = 0;
        shtmp[tid + tpb] = 0;
        if (ind < N / 2)
            shtmp[tid] = x[ind];
        if (ind + N / 2 < N)
            shtmp[tid + tpb] = x[ind + N / 2];

        for (int s = tpb; s > 0; s >>= 1) {
            if (tid < s) {
                shtmp[tid] += shtmp[tid + s];
            }
            item.barrier();
        }
        if (tid == 0) {
            sum[ibl] = shtmp[0]; // each block saves its partial result to an array
         /*
           sycl::atomic_ref<double, sycl::memory_order::relaxed,
                          sycl::memory_scope::device,
                          sycl::access::address_space::global_space>
              ref(sum[0]);
           ref.fetch_add(shtmp[0]);
         */
         // Alternatively, we could aggregate everything together at index 0.
         // Only useful when there not many partial sums left and when the device supports
         // atomic operations on FP64/double operands.
        }
    };
}
