from numba import cuda
import numpy as np
import time

# Define CUDA kernel with optimization
@cuda.jit
def optimized_kernel(data):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # Global thread ID
    if (idx % 32) == 0:  # Avoid warp divergence by aligning operations
        data[idx] *= 2  # Optimized operation

# Initialize data
data = np.arange(100, dtype=np.int32)
d_data = cuda.to_device(data)

# Configure threads and blocks
threads_per_block = 32
blocks_per_grid = (data.size + (threads_per_block - 1)) // threads_per_block

# Measure performance using Python timing
start_time = time.time()

# Launch kernel
optimized_kernel[blocks_per_grid, threads_per_block](d_data)

# Synchronize and measure execution time
cuda.synchronize()
execution_time = time.time() - start_time

# Copy data back to host and print results
result = d_data.copy_to_host()
print("Execution Time:", execution_time)
print("Result:", result)
