const static int tile_dim = 16;

__global__ void transpose_SM_nobc_kernel(float *in, float *out, int width, int height)
{
    __shared__ float tile[tile_dim][tile_dim + 1];

    int x_tile_index = blockIdx.x * tile_dim;
    int y_tile_index = blockIdx.y * tile_dim;

    int in_index =(y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
    int out_index =(x_tile_index + threadIdx.y) * height + (y_tile_index + threadIdx.x);

    tile[threadIdx.y][threadIdx.x] = in[in_index];

    __syncthreads();

    out[out_index] = tile[threadIdx.x][threadIdx.y];
}
