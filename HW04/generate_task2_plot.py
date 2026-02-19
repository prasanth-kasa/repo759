import subprocess
import matplotlib.pyplot as plt

n_values = [2**i for i in range(10, 30)]
R = 128
times_1024 = []
times_512 = [] 

print("Starting Task 2 scaling runs. This will take a while for large n...\n")

for n in n_values:
    print(f"--- Testing n=2^{n.bit_length()-1} ---")
    
    # Run task2 with 1024 threads
    result_1024 = subprocess.run(['./task2', str(n), str(R), '1024'], capture_output=True, text=True)
    raw_output_1024 = result_1024.stdout.strip()
    print(f"[Threads=1024] Raw C++ Output:\n{raw_output_1024}")
    
    time_1024 = float(raw_output_1024.split('\n')[-1])
    times_1024.append(time_1024)
    
    # Run task2 with 512 threads
    result_512 = subprocess.run(['./task2', str(n), str(R), '512'], capture_output=True, text=True)
    raw_output_512 = result_512.stdout.strip()
    print(f"[Threads=512] Raw C++ Output:\n{raw_output_512}\n")
    
    time_512 = float(raw_output_512.split('\n')[-1])
    times_512.append(time_512)

# Generate the plot directly in memory
plt.figure(figsize=(10, 6))
plt.plot(n_values, times_1024, marker='o', label='Threads = 1024')
plt.plot(n_values, times_512, marker='x', label='Threads = 512')

plt.title('Task 2: 1D Stencil Convolution Scaling (R=128)')
plt.xlabel('Array Size (n)')
plt.ylabel('Execution Time (ms)')
plt.xscale('log', base=2) 
plt.grid(True)
plt.legend()

# Save straight to PDF
plt.savefig('task2.pdf')
print("Done! Plot saved directly to task2.pdf.")