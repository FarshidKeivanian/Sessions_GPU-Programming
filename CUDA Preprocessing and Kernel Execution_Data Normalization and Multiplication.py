import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pycuda.compiler as compiler

# CUDA Kernel with preprocessing
mod = compiler.SourceModule("""
__global__ void preprocess_and_multiply(float *data, float factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 256) {
        // Preprocessing: Normalize data to [0, 1]
        data[idx] = data[idx] / 100.0;
        // Kernel operation: Multiply by factor
        data[idx] *= factor;
    }
}
""")

# Initialize data
data = np.random.uniform(0, 100, 256).astype(np.float32)
d_data = cuda.mem_alloc(data.nbytes)

# Copy data to GPU
cuda.memcpy_htod(d_data, data)

# Launch kernel
kernel = mod.get_function("preprocess_and_multiply")
kernel(d_data, np.float32(2.0), block=(256, 1, 1), grid=(1, 1))

# Copy back and print results
result = np.empty_like(data)
cuda.memcpy_dtoh(result, d_data)
print("Processed Data:", result)