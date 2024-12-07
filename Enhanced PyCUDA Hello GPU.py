import pycuda.autoinit  # Automatically manage CUDA context creation and cleanup
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# Define CUDA kernel code
kernel_code = """
__global__ void add_numbers(int a, int b, int *result) {
    // Perform addition and store the result
    *result = a + b;
}
"""

# Compile the kernel code
mod = SourceModule(kernel_code)

# Prepare the kernel function
add_numbers = mod.get_function("add_numbers")

# Define the two integers and allocate memory for the result
a = np.int32(5)
b = np.int32(7)
result_gpu = cuda.mem_alloc(np.int32().nbytes)  # Allocate GPU memory for the result

# Launch the kernel with parameters and retrieve the result
add_numbers(a, b, result_gpu, block=(1, 1, 1))

# Copy the result from GPU to CPU
result = np.empty_like(a)
cuda.memcpy_dtoh(result, result_gpu)  # Copy result from device to host

print("The sum of", a, "and", b, "is:", result)

# Free GPU memory
result_gpu.free()
