import matplotlib.pyplot as plt
import math

n_values = []
times = []

filename = 'task1_output.txt'

try:
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("Running for n ="):
               
                parts = line.split('(')
                n_str = parts[1].split(')')[0]
                current_n = int(n_str)
                
                if i + 1 < len(lines):
                    time_line = lines[i+1].strip()
                    try:
                        time_val = float(time_line)
                        
                        n_values.append(current_n)
                        times.append(time_val)
                    except ValueError:
                        print(f"Skipping invalid time value: {time_line}")
                
            i += 1

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, times, marker='o', linestyle='-', color='b', label='Scan Time')
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    plt.title('Scaling Analysis of Inclusive Scan (Task 1)')
    plt.xlabel('Array Size n')
    plt.ylabel('Time (ms)')
    plt.grid(True, which="both", ls="--")
    
    plt.savefig('task1.pdf')
    print("Plot saved as task1.pdf")

except FileNotFoundError:
    print(f"Error: Could not find file '{filename}'. Make sure the Slurm job has finished.")