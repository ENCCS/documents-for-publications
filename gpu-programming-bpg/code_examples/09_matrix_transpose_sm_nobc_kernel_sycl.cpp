auto transposeKernel(sycl::handler &cgh, const float *in, float *out, int width, int height)
{
    sycl::local_accessor<float, 1> tile{{tile_dim * (tile_dim + 1)}, cgh};
    return [=](sycl::nd_item<2> item)
    {
        int x_tile_index = item.get_group(1) * tile_dim;
        int y_tile_index = item.get_group(0) * tile_dim;
        int x_local_index = item.get_local_id(1);
        int y_local_index = item.get_local_id(0);
        int in_index = (y_tile_index + y_local_index) * width + (x_tile_index + x_local_index);
        int out_index = (x_tile_index + y_local_index) * width + (y_tile_index + x_local_index);

        tile[y_local_index * (tile_dim + 1) + x_local_index] = in[in_index];
        item.barrier();
        out[out_index] = tile[x_local_index * (tile_dim + 1) + y_local_index];
    };
}
