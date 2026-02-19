import matplotlib.pyplot as plt

# Your matrix sizes (2^5 to 2^14)
n_values = [2**i for i in range(5, 15)]

# TODO: Copy-paste the millisecond times from task1_results.txt here
times_1024 = [0.1, 0.2, 0.5, ...] 
# TODO: Copy-paste the times for your second thread block size here
times_256 = [0.15, 0.25, 0.6, ...] 

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(n_values, times_1024, marker='o', label='Threads = 1024')
plt.plot(n_values, times_256, marker='x', label='Threads = 256')

plt.title('Task 1: Matrix Multiplication Scaling')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Execution Time (ms)')
plt.grid(True)
plt.legend()

# Save as PDF exactly as requested by the assignment
plt.savefig('task1.pdf')
print("Plot saved as task1.pdf")