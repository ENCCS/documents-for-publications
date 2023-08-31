#define tpb 512 // size in this case has to be known at compile time

// this kernel has to be launched with at least N/2 threads
__global__ void reduction_one(double x, double *sum, int N)
{
    int ibl=blockIdx.y+blockIdx.x*gridDim.y;
    int ind=threadIdx.x+blockDim.x*ibl;

    __shared__ double shtmp[2 * tpb];
    shtmp[threadIdx.x] = 0; // for sums we initiate with 0, for other operations should be different
    if(ind < N / 2)
        shtmp[threadIdx.x] = x[ind];
    if(ind + N / 2 < N)
        shtmp[threadIdx.x+tpb] = x[ind+N/2];
    __syncthreads();
  
    for(int s = tpb; s > 0; s >>= 1)
    {
        if(threadIdx.x < s)
            shtmp[threadIdx.x] += shtmp[threadIdx.x + s];
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        sum[ibl]=shtmp[0]; // each block saves its partial result to an array
        // atomicAdd(&sum[0], shene[0]); // alternatively could aggregate everything together at index 0.
        //  Only use when there not many partial sums left
    }
}
