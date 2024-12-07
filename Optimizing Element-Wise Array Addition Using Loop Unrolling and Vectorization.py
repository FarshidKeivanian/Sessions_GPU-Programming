import numpy as np

# Example: Unrolling a loop for element-wise addition
N = 8
a = np.arange(N)
b = np.arange(N)

print("Array a:", a)
print("Array b:", b)

# Without unrolling (standard loop)
result = np.zeros(N)
for i in range(N):
    result[i] = a[i] + b[i]
print("\nResult using standard loop:")
print(result)

# With unrolling (manual implementation for small N)
result_unrolled = np.zeros(N)
result_unrolled[0] = a[0] + b[0]
result_unrolled[1] = a[1] + b[1]
result_unrolled[2] = a[2] + b[2]
result_unrolled[3] = a[3] + b[3]
result_unrolled[4] = a[4] + b[4]
result_unrolled[5] = a[5] + b[5]
result_unrolled[6] = a[6] + b[6]
result_unrolled[7] = a[7] + b[7]
print("\nResult using manual loop unrolling:")
print(result_unrolled)

# Using NumPy for vectorized operations (best practice in Python)
result_vectorized = a + b
print("\nResult using NumPy vectorized operation:")
print(result_vectorized)
