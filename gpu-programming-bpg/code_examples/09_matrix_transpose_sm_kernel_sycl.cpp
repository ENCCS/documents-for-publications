auto transposeKernel(sycl::handler &cgh, const float *in, float *out, int width, int height)
{
    sycl::local_accessor<float, 1> tile{{tile_dim * tile_dim}, cgh};
    return [=](sycl::nd_item<2> item)
    {
        int x_tile_index = item.get_group(1) * tile_dim;
        int y_tile_index = item.get_group(0) * tile_dim;
        int x_local_index = item.get_local_id(1);
        int y_local_index = item.get_local_id(0);
        int in_index = (y_tile_index + y_local_index) * width + (x_tile_index + x_local_index);
        int out_index = (x_tile_index + y_local_index) * width + (y_tile_index + x_local_index);

        tile[y_local_index * tile_dim + x_local_index] = in[in_index];
        item.barrier();
        out[out_index] = tile[x_local_index * tile_dim + y_local_index];
    };
}

/* Since allocating shared memory in SYCL requires sycl::handler, when calling parallel_for, an additional parameter must be passed:
 * cgh.parallel_for(kernel_range, transposeKernel(cgh, d_in, d_out, width, height));
 */
