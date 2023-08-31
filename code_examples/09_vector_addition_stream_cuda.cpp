// Distribute kernel for 'n_streams' streams, and record each stream's timing
for (int i = 0; i < n_streams; ++i)
{
    int offset = i * stream_size;

    // stamp the moment when the kernel is submitted on stream i
    cudaEventRecord(start_event[i], stream[i]); 
    cudaMemcpyAsync( &Ad[offset],  &Ah[offset], N/n_streams*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync( &Bd[offset],  &Bh[offset], N/n_streams*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
    
    //each call processes N/n_streams elements
    vector_add<<<gridsize / n_streams, blocksize, 0, stream[i]>>>(&Ad[offset], &Bd[offset], &Cd[offset], N/n_streams);
    cudaMemcpyAsync( &Ch[offset],  &Cd[offset], N/n_streams*sizeof(float), cudaMemcpyDeviceToHost, stream[i]);

    // stamp the moment when the kernel on stream i finished
    cudaEventRecord(stop_event[i], stream[i]); 
}
