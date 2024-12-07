import numpy as np
from numba import cuda

@cuda.jit
def matmul(A, B, C):
    row, col = cuda.grid(2)  # Get row and column index for each thread
    if row < C.shape[0] and col < C.shape[1]:
        temp = 0  # Temporary variable to store the computed value
        for k in range(A.shape[1]):  # Perform the dot product for the row and column
            temp += A[row, k] * B[k, col]
        C[row, col] = temp  # Assign the computed value to the result matrix

# Initialize matrices
N = 1024  # Size of the square matrices
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Transfer data to GPU
A_gpu = cuda.to_device(A)
B_gpu = cuda.to_device(B)
C_gpu = cuda.to_device(C)

# Define block and grid sizes
threads_per_block = (16, 16)  # Number of threads per block
blocks_per_grid = (N // 16, N // 16)  # Number of blocks per grid

# Launch the kernel
matmul[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_gpu)

# Copy result back to CPU
C = C_gpu.copy_to_host()

# Print a portion of the result
print("Resultant matrix (portion):")
print(C[:10, :10])  # Display the first 10x10 block of the result
