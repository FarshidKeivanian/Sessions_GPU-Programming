from numba import cuda
import numpy as np

# Check and initialize CUDA
if not cuda.is_available():
    raise RuntimeError("CUDA-capable GPU is not available.")

try:
    cuda.select_device(0)
    device_name = cuda.current_context().device.name
    print("Selected device:", device_name)

    # Check GPU memory info safely
    free_mem, total_mem = cuda.current_context().get_memory_info()
    print(f"Total GPU Memory: {total_mem / 1024**2:.2f} MB")
    print(f"Free GPU Memory: {free_mem / 1024**2:.2f} MB")

except Exception as e:
    print("Error during CUDA initialization:", e)
    raise RuntimeError("Failed to initialize CUDA.") from e

@cuda.jit
def matrix_reduction(mat, result):
    shared = cuda.shared.array(shape=(256,), dtype=np.float32)  # Shared memory
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    
    # Load data into shared memory
    idx = bid * bdim + tid
    if idx < mat.size:
        shared[tid] = mat[idx]
    else:
        shared[tid] = 0  # Handle out-of-bounds threads
    cuda.syncthreads()
    
    # Perform parallel reduction
    stride = bdim // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()
        stride //= 2
    
    # Write result of this block to the result array
    if tid == 0:
        result[bid] = shared[0]
        
# Initialize data
N = 512  # Reduce size for debugging
data = np.random.rand(N).astype(np.float32)
result = np.zeros((N // 128,), dtype=np.float32)

# Copy to device
threads_per_block = 128
blocks = N // threads_per_block

try:
    d_data = cuda.to_device(data)
    print("Data copied to device successfully.")
except Exception as e:
    print("Error in copying data to device:", e)
    raise

try:
    d_result = cuda.to_device(result)
    print("Result array copied to device successfully.")
except Exception as e:
    print("Error in copying result to device:", e)
    raise

# Call kernel
try:
    matrix_reduction[blocks, threads_per_block](d_data, d_result)
except Exception as e:
    print("Error during kernel execution:", e)
    raise

# Retrieve and display result
try:
    result_host = d_result.copy_to_host()
    total_sum = np.sum(result_host)
    print("Total Sum:", total_sum)
except Exception as e:
    print("Error in copying result back to host:", e)
    raise
