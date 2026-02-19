import subprocess
import matplotlib.pyplot as plt

n_values = [2**i for i in range(5, 15)]
times_1024 = []
times_256 = []  

print("Starting Task 1 scaling runs. This might take a minute...\n")

for n in n_values:
    print(f"--- Testing n={n} ---")
    
    # Run task1 with 1024 threads
    result_1024 = subprocess.run(['./task1', str(n), '1024'], capture_output=True, text=True)
    raw_output_1024 = result_1024.stdout.strip()
    print(f"[Threads=1024] Raw C++ Output:\n{raw_output_1024}")
    
    time_1024 = float(raw_output_1024.split('\n')[-1])
    times_1024.append(time_1024)
    
    # Run task1 with 256 threads
    result_256 = subprocess.run(['./task1', str(n), '256'], capture_output=True, text=True)
    raw_output_256 = result_256.stdout.strip()
    print(f"[Threads=256] Raw C++ Output:\n{raw_output_256}\n")
    
    time_256 = float(raw_output_256.split('\n')[-1])
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