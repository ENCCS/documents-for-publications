// Distribute kernel for 'n_streams' streams, and record each stream's timing
for (int i = 0; i < n_streams; ++i)
{
    int offset = i * (N/stream_size);
    
    // stamp the moment when the kernel is submitted on stream i
    hipEventRecord(start_event[i], stream[i]);
    hipMemcpy(d_in, matrix_in.data(), width * height * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpyAsync( &Ad[offset],  &Ah[offset], N/n_streams*sizeof(float), hipMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync( &Bd[offset],  &Bh[offset], N/n_streams*sizeof(float), hipMemcpyHostToDevice, stream[i]);
  
    // each call processes N/n_streams elements
    vector_add<<<gridsize / n_streams, blocksize, 0, stream[i]>>>(&Ad[offset], &Bd[offset], &Cd[offset], N/n_streams);
    hipMemcpyAsync( &Ch[offset],  &Cd[offset], N/n_streams*sizeof(float), hipMemcpyDeviceToHost, stream[i]);

    // stamp the moment when the kernel on stream i finished
    hipEventRecord(stop_event[i], stream[i]); 
}
