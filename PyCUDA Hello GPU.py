import pycuda.autoinit  # Automatically manage CUDA context creation and cleanup
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Define CUDA kernel code
kernel_code = """
__global__ void hello_world(char *message) {
    // Only one thread will execute this, as a demonstration
    printf("%s\\n", message);
}
"""

# Compile the kernel code
mod = SourceModule(kernel_code)

# Prepare the kernel function
hello_world = mod.get_function("hello_world")

# Create a message to send to the GPU
message = "Hello from GPU!".encode("utf-8")  # Encoding string to bytes
message_gpu = cuda.mem_alloc(len(message))   # Allocate GPU memory for the message
cuda.memcpy_htod(message_gpu, message)       # Copy the message to GPU memory

# Launch the kernel (1 block with 1 thread, for simplicity)
hello_world(message_gpu, block=(1, 1, 1))

# Free GPU memory
message_gpu.free()
