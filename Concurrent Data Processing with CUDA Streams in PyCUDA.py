import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

# Define the CUDA kernel
mod = SourceModule("""
__global__ void kernel(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] * 2;  // Example operation: double the values
}
""")
kernel = mod.get_function("kernel")

# Host data
data_chunk1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
data_chunk2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

# Allocate device memory
d_chunk1 = cuda.mem_alloc(data_chunk1.nbytes)
d_chunk2 = cuda.mem_alloc(data_chunk2.nbytes)

# Create CUDA streams
stream1 = cuda.Stream()
stream2 = cuda.Stream()

# Copy data to device asynchronously
cuda.memcpy_htod_async(d_chunk1, data_chunk1, stream1)
cuda.memcpy_htod_async(d_chunk2, data_chunk2, stream2)

# Launch kernels asynchronously
block_size = 4  # Number of threads per block
grid_size = 1   # Number of blocks

kernel(d_chunk1, block=(block_size, 1, 1), grid=(grid_size, 1), stream=stream1)
kernel(d_chunk2, block=(block_size, 1, 1), grid=(grid_size, 1), stream=stream2)

# Copy results back to host asynchronously
cuda.memcpy_dtoh_async(data_chunk1, d_chunk1, stream1)
cuda.memcpy_dtoh_async(data_chunk2, d_chunk2, stream2)

# Synchronize streams
stream1.synchronize()
stream2.synchronize()

# Display results
print("Chunk 1 processed:", data_chunk1)
print("Chunk 2 processed:", data_chunk2)
