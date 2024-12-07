import numpy as np
from numba import cuda

# Define kernel
@cuda.jit
def add_arrays(a, b, result):
    idx = cuda.grid(1)  # Get global thread index
    if idx < a.size:  # Boundary check
        result[idx] = a[idx] + b[idx]

# Driver code
N = 100000  # Size of the arrays
A = np.ones(N, dtype=np.float32)
B = np.ones(N, dtype=np.float32)
C = np.zeros(N, dtype=np.float32)

# Allocate arrays on the device
A_device = cuda.to_device(A)
B_device = cuda.to_device(B)
C_device = cuda.device_array(N, dtype=np.float32)

# Launch kernel
threads_per_block = 1024
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
add_arrays[blocks_per_grid, threads_per_block](A_device, B_device, C_device)

# Copy result back to host
C = C_device.copy_to_host()

# Print a sample of the result
print("First 10 elements of result:", C[:10])
