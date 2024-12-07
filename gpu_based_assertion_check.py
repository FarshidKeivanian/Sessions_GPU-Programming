import pycuda.driver as cuda
import pycuda.compiler as nvcc
import pycuda.autoinit
import numpy as np

# CUDA Kernel
kernel_code = """
__global__ void check_assert(int *data) {
    int idx = threadIdx.x;
    printf("Thread %d is checking value %d\\n", idx, data[idx]);
    if (data[idx] < 0) {
        printf("Assertion failed at index %d: value %d\\n", idx, data[idx]);
    }
}
"""

# Compile the kernel
module = nvcc.SourceModule(kernel_code, options=["--ptxas-options=-v"], keep=True)
check_assert = module.get_function("check_assert")

# Prepare data
data = np.array([1, 2, -3], dtype=np.int32)
data_gpu = cuda.mem_alloc(data.nbytes)
cuda.memcpy_htod(data_gpu, data)

# Launch kernel
try:
    check_assert(data_gpu, block=(3, 1, 1))
    cuda.Context.synchronize()  # Ensures execution completes
    print("Kernel execution completed successfully.")
except cuda.Error as e:
    print(f"CUDA Error: {e}")
