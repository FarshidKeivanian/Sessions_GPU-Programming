import cupy as cp

try:
    # Test CuPy functionality
    arr = cp.arange(10)
    print("CuPy array created:", arr)
    
    # Main reduction logic
    def reduce_sum(data):
        return cp.sum(data)

    n = 1024
    data = cp.arange(1, n + 1, dtype=cp.int32)
    result = reduce_sum(data)
    print("Sum of numbers from 1 to", n, "is", result.get())
except AttributeError as e:
    print("AttributeError:", e)
    print("Check your CuPy installation and compatibility with CUDA.")
except Exception as e:
    print("Error occurred:", e)
