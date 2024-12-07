import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# Initialize matrices
a = np.random.randn(4, 4).astype(np.float32)
b = np.random.randn(4, 4).astype(np.float32)
c = np.zeros_like(a)

# Allocate memory on the GPU
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Copy data to GPU
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Define GPU kernel
mod = SourceModule("""
__global__ void add_matrix(float *a, float *b, float *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * width + col;

    if (row < width && col < width) {
        c[idx] = a[idx] + b[idx];
    }
}
""")

add_matrix = mod.get_function("add_matrix")

# Execute kernel (using grid and block configuration)
matrix_size = 4  # Size of the matrix
block_size = 2  # Number of threads per block (2x2 threads)
grid_size = (matrix_size // block_size, matrix_size // block_size, 1)

add_matrix(a_gpu, b_gpu, c_gpu, np.int32(matrix_size),
           block=(block_size, block_size, 1),
           grid=grid_size)

# Retrieve result
cuda.memcpy_dtoh(c, c_gpu)
print("Matrix A:\n", a)
print("Matrix B:\n", b)
print("Result (A + B):\n", c)
