import cupy as cp
a = cp.array([1, 2, 3, 4, 5])
b = cp.array([10, 20, 30, 40, 50])
c = a * b  # Element-wise multiplication
print(c)  # Should print the element-wise multiplication result
