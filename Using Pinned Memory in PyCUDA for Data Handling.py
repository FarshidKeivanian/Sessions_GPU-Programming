import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Allocate pinned memory
data = np.array([1, -2, 3, -4, 5], dtype=np.int32)
pinned_memory = cuda.pagelocked_empty(len(data), dtype=np.int32)
pinned_memory[:] = data[:]

# Show the original and pinned memory content
print("Original data:", data)
print("Pinned memory data:", pinned_memory)
