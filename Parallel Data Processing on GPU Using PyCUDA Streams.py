import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pycuda.compiler as compiler

# CUDA Kernel
mod = compiler.SourceModule("""
__global__ void multiply(int *data, int factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] *= factor;
}
""")

# Initialize data
N = 256
data1 = np.arange(N, dtype=np.int32)
data2 = np.arange(N, dtype=np.int32)

# Allocate memory on GPU
data1_gpu = cuda.mem_alloc(data1.nbytes)
data2_gpu = cuda.mem_alloc(data2.nbytes)

# Copy data to GPU
cuda.memcpy_htod(data1_gpu, data1)
cuda.memcpy_htod(data2_gpu, data2)

# Create streams
stream1 = cuda.Stream()
stream2 = cuda.Stream()

# Get kernel function
multiply_kernel = mod.get_function("multiply")

# Launch kernels in different streams
multiply_kernel(data1_gpu, np.int32(2), block=(256, 1, 1), grid=(1, 1), stream=stream1)
multiply_kernel(data2_gpu, np.int32(3), block=(256, 1, 1), grid=(1, 1), stream=stream2)

# Add callbacks (optional, simulating with sync here for simplicity)
stream1.synchronize()
stream2.synchronize()

# Copy results back to host
result1 = np.empty_like(data1)
result2 = np.empty_like(data2)
cuda.memcpy_dtoh(result1, data1_gpu)
cuda.memcpy_dtoh(result2, data2_gpu)

# Print results
print("Result from Stream 1:", result1)
print("Result from Stream 2:", result2)
