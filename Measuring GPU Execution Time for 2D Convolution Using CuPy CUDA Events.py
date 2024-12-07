def convolution_2d(input_tensor, kernel, stride=1, padding=0):
    """
    Perform 2D convolution using CuPy.
    Args:
        input_tensor (ndarray): 3D input tensor (batch_size, height, width).
        kernel (ndarray): 2D kernel (height, width).
        stride (int): Stride of the convolution.
        padding (int): Padding size around the input.
    Returns:
        ndarray: The output tensor after the convolution operation.
    """
    if padding > 0:
        input_tensor = cp.pad(input_tensor, ((0, 0), (padding, padding), (padding, padding)), mode='constant')

    batch_size, in_height, in_width = input_tensor.shape
    kernel_height, kernel_width = kernel.shape
    out_height = (in_height - kernel_height) // stride + 1
    out_width = (in_width - kernel_width) // stride + 1

    output = cp.zeros((batch_size, out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            region = input_tensor[:, i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
            output[:, i, j] = cp.sum(region * kernel, axis=(1, 2))

    return output

# Main program
if __name__ == "__main__":
    # Create a random input tensor (batch_size=1, height=5, width=5)
    input_tensor = cp.random.rand(1, 5, 5).astype(cp.float32)

    # Create a random 2D kernel (3x3)
    kernel = cp.random.rand(3, 3).astype(cp.float32)

    # Initialize CUDA events
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    # Measure execution time
    start.record()
    output = convolution_2d(input_tensor, kernel)
    end.record()

    # Synchronize and compute elapsed time
    end.synchronize()
    elapsed_time = cp.cuda.get_elapsed_time(start, end)

    # Display results
    print("Execution Time (ms):", elapsed_time)
    print("\nOutput Tensor:\n", output.get())  # Transfer output back to CPU for display