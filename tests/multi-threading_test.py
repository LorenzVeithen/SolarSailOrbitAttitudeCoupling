import matplotlib.pyplot as plt
import matplotlib
import threading
import time
matplotlib.pyplot.switch_backend('Agg')

def generate_and_save_plot(index):
    # Simulating some computation
    x = range(100)
    y = [i**index for i in x]

    print(f'Plot {index} saved.')
    time.sleep(10)

# List to keep track of thread objects
threads = []

# Creating and starting threads
t0 = time.time()
for i in range(8):
    thread = threading.Thread(target=generate_and_save_plot, args=(i,))
    thread.start()
    threads.append(thread)

# Waiting for all threads to complete
for thread in threads:
    thread.join()
print(time.time()-t0)

print("All plots generated and saved.")
