{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8fb26935-0f96-449f-8fcb-609b3c1fc5a4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GPU Total Sum: 51278.734375\n",
      "CPU Total Sum: 51278.734375\n",
      "Difference: 0.0\n"
     ]
    }
   ],
   "source": [
    "from numba import cuda\n",
    "import numpy as np\n",
    "\n",
    "@cuda.jit\n",
    "def parallel_reduction(input, output, size):\n",
    "    # Allocate shared memory for the block\n",
    "    shared_data = cuda.shared.array(256, dtype=np.float32)\n",
    "    \n",
    "    tid = cuda.threadIdx.x\n",
    "    block_idx = cuda.blockIdx.x\n",
    "    grid_stride = cuda.gridDim.x * cuda.blockDim.x\n",
    "    \n",
    "    # Each thread processes multiple elements, reducing them into shared memory\n",
    "    local_sum = 0.0\n",
    "    idx = cuda.grid(1)  # Global thread index\n",
    "\n",
    "    # Stride loop to handle arrays larger than the grid size\n",
    "    while idx < size:\n",
    "        local_sum += input[idx]\n",
    "        idx += grid_stride\n",
    "\n",
    "    # Store local sum into shared memory\n",
    "    shared_data[tid] = local_sum\n",
    "    cuda.syncthreads()\n",
    "\n",
    "    # Reduce within shared memory (minimized warp divergence)\n",
    "    stride = cuda.blockDim.x // 2\n",
    "    while stride > 0:\n",
    "        if tid < stride:\n",
    "            shared_data[tid] += shared_data[tid + stride]\n",
    "        cuda.syncthreads()\n",
    "        stride //= 2\n",
    "\n",
    "    # Write result of each block to the output array\n",
    "    if tid == 0:\n",
    "        output[block_idx] = shared_data[0]\n",
    "\n",
    "def main():\n",
    "    size = 102400  # Large input size for better GPU utilization\n",
    "    threads_per_block = 256\n",
    "    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block\n",
    "\n",
    "    # Input array (e.g., random values)\n",
    "    h_input = np.random.rand(size).astype(np.float32)\n",
    "\n",
    "    # Allocate output array to hold block results\n",
    "    h_output = np.zeros(blocks_per_grid, dtype=np.float32)\n",
    "\n",
    "    # Allocate device memory\n",
    "    d_input = cuda.to_device(h_input)\n",
    "    d_output = cuda.device_array(blocks_per_grid, dtype=np.float32)\n",
    "\n",
    "    # Launch kernel\n",
    "    parallel_reduction[blocks_per_grid, threads_per_block](d_input, d_output, size)\n",
    "\n",
    "    # Copy partial results back to host\n",
    "    d_output.copy_to_host(h_output)\n",
    "\n",
    "    # Perform final reduction on CPU\n",
    "    total_sum = np.sum(h_output)\n",
    "\n",
    "    # Validate results\n",
    "    expected_sum = np.sum(h_input)\n",
    "    print(f\"GPU Total Sum: {total_sum}\")\n",
    "    print(f\"CPU Total Sum: {expected_sum}\")\n",
    "    print(f\"Difference: {abs(total_sum - expected_sum)}\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
