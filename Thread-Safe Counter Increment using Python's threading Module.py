import threading

# Shared counter
counter = 0

# Lock for thread safety
counter_lock = threading.Lock()

def safe_increment():
    global counter
    with counter_lock:
        counter += 1

# Simulating parallel execution
threads = []
for _ in range(10):  # Launch 10 threads
    thread = threading.Thread(target=safe_increment)
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print("Final Counter Value:", counter)
