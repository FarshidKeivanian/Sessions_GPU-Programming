def max_pooling(input_tensor, pool_size, stride):
    batch_size, in_height, in_width = input_tensor.shape
    out_height = (in_height - pool_size) // stride + 1
    out_width = (in_width - pool_size) // stride + 1

    # Initialize output tensor
    output = cp.zeros((batch_size, out_height, out_width))

    # Perform pooling
    for i in range(out_height):
        for j in range(out_width):
            region = input_tensor[:, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[:, i, j] = cp.max(region, axis=(1, 2))

    return output
