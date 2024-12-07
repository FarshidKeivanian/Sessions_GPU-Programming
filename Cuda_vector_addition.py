import cupy as cp

# Test CuPy with a simple array operation
a = cp.array([1, 2, 3, 4, 5])
b = cp.array([10, 20, 30, 40, 50])
c = a + b
print(c)  # Should print the element-wise addition result if CuPy works correctly
