import pycuda.driver as cuda
import pycuda.autoinit

try:
    # Allocate too much memory to trigger an error
    large_mem = cuda.mem_alloc(10**12)
except cuda.Error as e:
    print(f"CUDA Error: {e}")
