import subprocess
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

powers = range(10, 30)
ns = [2**p for p in powers]

times_512 = []
times_16 = []

print("Starting benchmark runs...")

for n in ns:
    res_512 = subprocess.run(["./task3", str(n), "512"], capture_output=True, text=True)
    t_512 = float(res_512.stdout.split('\n')[0])
    times_512.append(t_512)

    res_16 = subprocess.run(["./task3", str(n), "16"], capture_output=True, text=True)
    t_16 = float(res_16.stdout.split('\n')[0])
    times_16.append(t_16)
    
    print(f"Finished n=2^{int(n).bit_length()-1}")

plt.figure(figsize=(10, 6))
plt.plot(ns, times_512, 'b-o', label='512 threads/block')
plt.plot(ns, times_16, 'r-x', label='16 threads/block')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Array Size (n)')
plt.ylabel('Time (ms)')
plt.title('Task 3: Vscale Execution Time vs Array Size')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.savefig('task3.pdf')
print("Plot saved to task3.pdf")
