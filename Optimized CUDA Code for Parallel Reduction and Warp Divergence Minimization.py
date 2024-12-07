from numba import cuda
import numpy as np

@cuda.jit
def parallel_reduction(input, output, size):
    # Allocate shared memory for the block
    shared_data = cuda.shared.array(256, dtype=np.float32)
    
    tid = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    grid_stride = cuda.gridDim.x * cuda.blockDim.x
    
    # Each thread processes multiple elements, reducing them into shared memory
    local_sum = 0.0
    idx = cuda.grid(1)  # Global thread index

    # Stride loop to handle arrays larger than the grid size
    while idx < size:
        local_sum += input[idx]
        idx += grid_stride

    # Store local sum into shared memory
    shared_data[tid] = local_sum
    cuda.syncthreads()

    # Reduce within shared memory (minimized warp divergence)
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared_data[tid] += shared_data[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # Write result of each block to the output array
    if tid == 0:
        output[block_idx] = shared_data[0]

def main():
    size = 102400  # Large input size for better GPU utilization
    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

    # Input array (e.g., random values)
    h_input = np.random.rand(size).astype(np.float32)

    # Allocate output array to hold block results
    h_output = np.zeros(blocks_per_grid, dtype=np.float32)

    # Allocate device memory
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(blocks_per_grid, dtype=np.float32)

    # Launch kernel
    parallel_reduction[blocks_per_grid, threads_per_block](d_input, d_output, size)

    # Copy partial results back to host
    d_output.copy_to_host(h_output)

    # Perform final reduction on CPU
    total_sum = np.sum(h_output)

    # Validate results
    expected_sum = np.sum(h_input)
    print(f"GPU Total Sum: {total_sum}")
    print(f"CPU Total Sum: {expected_sum}")
    print(f"Difference: {abs(total_sum - expected_sum)}")

if __name__ == "__main__":
    main()
