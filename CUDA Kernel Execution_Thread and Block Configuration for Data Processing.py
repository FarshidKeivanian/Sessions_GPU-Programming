from numba import cuda
import numpy as np

# Define CUDA kernel
@cuda.jit
def simple_kernel(data):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # Global thread ID
    if idx < data.size:  # Check bounds
        data[idx] += 10  # Perform operation

# Initialize data
data = np.arange(100, dtype=np.int32)
d_data = cuda.to_device(data)

# Configure threads and blocks
threads_per_block = 32
blocks_per_grid = (data.size + (threads_per_block - 1)) // threads_per_block

# Launch kernel
simple_kernel[blocks_per_grid, threads_per_block](d_data)

# Copy data back to host and print results
result = d_data.copy_to_host()
print(result)
