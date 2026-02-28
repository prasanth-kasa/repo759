import matplotlib.pyplot as plt
import re

def parse_task1(filename):
    n_vals = []
    times_int, times_float, times_double = [], [], []
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return [], [], [], []

    current_n = None
    temp_vals = []
    
    for line in lines:
        # Look for the echo statement from the bash script
        match = re.search(r'n=(\d+)', line)
        if match:
            if current_n is not None and len(temp_vals) >= 9:
                n_vals.append(current_n)
                times_int.append(temp_vals[2])     # 3rd printed value is int time
                times_float.append(temp_vals[5])   # 6th printed value is float time
                times_double.append(temp_vals[8])  # 9th printed value is double time
            current_n = int(match.group(1))
            temp_vals = []
        else:
            try:
                val = float(line.strip())
                temp_vals.append(val)
            except ValueError:
                continue

    if current_n is not None and len(temp_vals) >= 9:
        n_vals.append(current_n)
        times_int.append(temp_vals[2])
        times_float.append(temp_vals[5])
        times_double.append(temp_vals[8])

    return n_vals, times_int, times_float, times_double

def parse_task2(filename):
    n_vals_1024, times_1024 = [], []
    n_vals_other, times_other = [], []
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return [], [], [], [], None

    current_n = None
    current_threads = None
    other_thread_count = None
    temp_vals = []

    for line in lines:
        match = re.search(r'N=(\d+), threads=(\d+)', line)
        if match:
            if current_n is not None and current_threads is not None and len(temp_vals) >= 2:
                if current_threads == 1024:
                    n_vals_1024.append(current_n)
                    times_1024.append(temp_vals[1]) # 2nd printed value is time
                else:
                    other_thread_count = current_threads
                    n_vals_other.append(current_n)
                    times_other.append(temp_vals[1])
            
            current_n = int(match.group(1))
            current_threads = int(match.group(2))
            temp_vals = []
        else:
            try:
                val = float(line.strip())
                temp_vals.append(val)
            except ValueError:
                continue

    if current_n is not None and len(temp_vals) >= 2:
        if current_threads == 1024:
            n_vals_1024.append(current_n)
            times_1024.append(temp_vals[1])
        else:
            n_vals_other.append(current_n)
            times_other.append(temp_vals[1])

    return n_vals_1024, times_1024, n_vals_other, times_other, other_thread_count

def main():
    # --- Task 1 Plot ---
    n1, t_int, t_float, t_double = parse_task1('task1_results.txt')
    if n1:
        plt.figure(figsize=(10, 6))
        plt.plot(n1, t_int, marker='o', label='int (matmul_1)')
        plt.plot(n1, t_float, marker='s', label='float (matmul_2)')
        plt.plot(n1, t_double, marker='^', label='double (matmul_3)')
        plt.xlabel('Matrix Dimension (n)')
        plt.ylabel('Time (ms)')
        plt.title('Task 1: Tiled Matrix Multiplication Performance')
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.savefig('task1_plot.png')
        print("Saved task1_plot.png")

    # --- Task 2 Plot ---
    n2_1024, t_1024, n2_other, t_other, other_threads = parse_task2('task2_results.txt')
    if n2_1024 and n2_other:
        plt.figure(figsize=(10, 6))
        plt.plot(n2_1024, t_1024, marker='o', label='threads_per_block = 1024')
        plt.plot(n2_other, t_other, marker='s', label=f'threads_per_block = {other_threads}')
        plt.xlabel('Array Length (N)')
        plt.ylabel('Time (ms)')
        plt.title('Task 2: Parallel Reduction Performance')
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.savefig('task2_plot.png')
        print("Saved task2_plot.png")

if __name__ == '__main__':
    main()