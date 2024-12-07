import cupy as cp

def convolution_2d(input_tensor, kernel, stride=1, padding=0):
    # Add padding to the input tensor
    if padding > 0:
        input_tensor = cp.pad(input_tensor, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    # Compute output dimensions
    batch_size, in_height, in_width = input_tensor.shape
    kernel_height, kernel_width = kernel.shape
    out_height = (in_height - kernel_height) // stride + 1
    out_width = (in_width - kernel_width) // stride + 1

    # Initialize output tensor
    output = cp.zeros((batch_size, out_height, out_width))

    # Perform convolution
    for i in range(out_height):
        for j in range(out_width):
            region = input_tensor[:, i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
            output[:, i, j] = cp.sum(region * kernel, axis=(1, 2))

    return output
