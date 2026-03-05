import pandas as pd
import matplotlib.pyplot as plt

def plot_task1():
    try:
        df1 = pd.read_csv('task1_data.csv')
        plt.figure(figsize=(8, 5))
        plt.plot(df1['n'], df1['time'], marker='o', linestyle='-', color='b')
        plt.xlabel('Matrix Dimension (n)')
        plt.ylabel('Time (milliseconds)')
        plt.title('Task 1: cuBLAS Matrix Multiplication Time vs. n')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('task1.pdf')
        print("Generated task1.pdf successfully.")
    except FileNotFoundError:
        print("task1_data.csv not found. Did the Slurm job finish?")

def plot_task2():
    try:
        df2 = pd.read_csv('task2_data.csv')
        plt.figure(figsize=(8, 5))
        plt.plot(df2['n'], df2['time'], marker='o', linestyle='-', color='r')
        plt.xlabel('Array Size (n)')
        plt.ylabel('Time (milliseconds)')
        plt.title('Task 2: Hillis-Steele Scan Time vs. n')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('task2.pdf')
        print("Generated task2.pdf successfully.")
    except FileNotFoundError:
        print("task2_data.csv not found. Did the Slurm job finish?")

if __name__ == '__main__':
    plot_task1()
    plot_task2()