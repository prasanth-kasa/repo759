import matplotlib.pyplot as plt

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

    if not n_values:
        print("No valid data found.")
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(n_values, times, marker='o', linestyle='-', color='b', label='Scan Time')
        
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        
        plt.title('Scaling Analysis (Log Scale)')
        plt.xlabel('Array Size n')
        plt.ylabel('Time (ms)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.savefig('task1_log.pdf')
        print("Saved task1_log.pdf")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(n_values, times, marker='o', linestyle='-', color='r', label='Scan Time')
        
        plt.xscale('linear')
        plt.yscale('linear')
        
        plt.title('Scaling Analysis (Linear Scale)')
        plt.xlabel('Array Size n')
        plt.ylabel('Time (ms)')
        plt.grid(True)
        
        plt.savefig('task1_linear.pdf')
        print("Saved task1_linear.pdf")
        plt.close()

except FileNotFoundError:
    print(f"Error: Could not find file '{filename}'. Make sure the Slurm job has finished.")