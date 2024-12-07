import numpy as np
import threading
from threading import Lock

# Data array and histogram
data = np.random.randint(0, 10, size=100)  # Random integers between 0 and 9
histogram = np.zeros(10, dtype=int)

# Lock for thread safety
hist_lock = Lock()

def update_histogram(value):
    with hist_lock:
        histogram[value] += 1

# Simulating parallel computation
threads = []
for value in data:
    thread = threading.Thread(target=update_histogram, args=(value,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print("Histogram:", histogram)
