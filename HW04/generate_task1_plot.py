import subprocess
import matplotlib.pyplot as plt

# The matrix sizes specified in the assignment
n_values = [2**i for i in range(5, 15)]
times_1024 = []
times_256 = []  # Our chosen second block size

print("Starting Task 1 scaling runs. This might take a minute...")

for n in n_values:
    # Run task1 with 1024 threads
    print(f"Testing n={n} with 1024 threads...")
    result_1024 = subprocess.run(['./task1', str(n), '1024'], capture_output=True, text=True)
    # Extract the last line of output (the time)
    time_1024 = float(result_1024.stdout.strip().split('\n')[-1])
    times_1024.append(time_1024)
    
    # Run task1 with 256 threads
    print(f"Testing n={n} with 256 threads...")
    result_256 = subprocess.run(['./task1', str(n), '256'], capture_output=True, text=True)
    time_256 = float(result_256.stdout.strip().split('\n')[-1])
    times_256.append(time_256)

# Generate the plot directly in memory
plt.figure(figsize=(10, 6))
plt.plot(n_values, times_1024, marker='o', label='Threads = 1024')
plt.plot(n_values, times_256, marker='x', label='Threads = 256')

plt.title('Task 1: Matrix Multiplication Scaling')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Execution Time (ms)')
plt.grid(True)
plt.legend()

# Save straight to PDF
plt.savefig('task1.pdf')
print("Done! Plot saved directly to task1.pdf.")