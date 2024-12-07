import cupy as cp

def parallel_preprocessing(input_data, stream):
    with stream:
        return cp.sqrt(input_data)
