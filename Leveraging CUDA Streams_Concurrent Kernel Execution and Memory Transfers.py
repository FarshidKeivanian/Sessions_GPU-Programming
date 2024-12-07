import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pycuda.compiler as compiler

# CUDA Kernel
mod = compiler.SourceModule("""
__global__ void add(int *a, int *b, int *c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}
""")

# Initialize data
N = 256
a = np.arange(N).astype(np.int32)
b = np.arange(N).astype(np.int32)
c = np.zeros(N).astype(np.int32)

# Allocate memory on GPU
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Copy data to GPU
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Create streams
stream1 = cuda.Stream()
stream2 = cuda.Stream()

# Launch kernel on stream1
add_kernel = mod.get_function("add")
add_kernel(a_gpu, b_gpu, c_gpu, block=(256, 1, 1), grid=(1, 1), stream=stream1)

# Copy result back asynchronously using stream2
cuda.memcpy_dtoh_async(c, c_gpu, stream2)

# Synchronize streams
stream1.synchronize()
stream2.synchronize()

# Print result
print("Result:", c)
