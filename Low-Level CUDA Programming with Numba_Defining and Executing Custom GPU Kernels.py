from numba import cuda
import numpy as np

@cuda.jit
def add_arrays(a, b, result):
    i = cuda.grid(1)
    if i < a.size:
        result[i] = a[i] + b[i]

# Initialize arrays
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([10, 20, 30, 40], dtype=np.float32)
result = np.zeros_like(a)

# Transfer to device and execute kernel
add_arrays[1, len(a)](a, b, result)
print("Result:", result)
