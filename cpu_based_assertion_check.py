import numpy as np

# Define the equivalent function
def check_assert(data):
    for idx, value in enumerate(data):
        if value < 0:
            print(f"Assertion failed at index {idx}: value {value}")

# Prepare data
data = np.array([1, 2, -3], dtype=np.int32)

# Call the function
check_assert(data)