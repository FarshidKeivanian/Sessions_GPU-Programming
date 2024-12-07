import cupy as cp

def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

n = 5
result = factorial_recursive(n)
print(f"Factorial of {n} is {result}")
