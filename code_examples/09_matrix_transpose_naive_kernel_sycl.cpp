auto transposeKernel(const float *in, float *out, int width, int height)
{
    return [=](sycl::nd_item<2> item)
    {
        int x_index = item.get_global_id(1);
        int y_index = item.get_global_id(0);
        int in_index = y_index * width + x_index;
        int out_index = x_index * height + y_index;
        out[out_index] = in[in_index];
    };
}