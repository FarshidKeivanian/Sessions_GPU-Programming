import numpy as np
from numba import cuda

# Define matrix dimensions
N = 512  # Assume a square matrix of N x N

# Kernel for optimized matrix multiplication with shared memory
@cuda.jit
def optimized_matrix_mult(A, B, C):
    # Define shared memory arrays with np.float32
    shared_A = cuda.shared.array((16, 16), dtype=np.float32)
    shared_B = cuda.shared.array((16, 16), dtype=np.float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x  # Block width

    row = by * bw + ty
    col = bx * bw + tx

    temp_sum = 0.0
    for i in range(0, N, bw):
        # Load data into shared memory
        shared_A[ty, tx] = A[row, i + tx]
        shared_B[ty, tx] = B[i + ty, col]

        # Synchronize threads within the block
        cuda.syncthreads()

        # Perform computation on shared data
        for j in range(bw):
            temp_sum += shared_A[ty, j] * shared_B[j, tx]

        # Synchronize again before loading new data
        cuda.syncthreads()

    if row < N and col < N:
        C[row, col] = temp_sum


# Initialize matrices
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Transfer to GPU
A_global = cuda.to_device(A)
B_global = cuda.to_device(B)
C_global = cuda.to_device(C)

# Configure grid and block sizes
threads_per_block = (16, 16)
blocks_per_grid = (N // threads_per_block[0], N // threads_per_block[1])

# Launch kernel
optimized_matrix_mult[blocks_per_grid, threads_per_block](A_global, B_global, C_global)

# Copy the result back to the host
C_result = C_global.copy_to_host()

print("Matrix multiplication with shared memory completed successfully.")
