
from pathlib import Path

import json
import sys
import matplotlib.pyplot as plt


experiment_path = Path("/raid/shared/llm-inference-scaling/prefix_sampling_experiments")

# Function to plot the number of problems solved over time
def plot_problems_solved(experiment_name):
    try:
        # Load JSON data from files
        ex_path = experiment_path / experiment_name
        file_path_prefix = ex_path / "times_prefix_sampling.json"
        file_path_baseline = ex_path / "times_baseline.json"
        with open(file_path_prefix, 'r') as file:
            data_prefix = json.load(file)

        with open(file_path_baseline, 'r') as file:
            data_baseline = json.load(file)

        # Extract and sort data for prefix sampling
        problems_prefix = list(data_prefix.keys())
        times_prefix = list(data_prefix.values())
        sorted_data_prefix = sorted(zip(times_prefix, problems_prefix))
        sorted_times_prefix = [item[0] for item in sorted_data_prefix]
        cumulative_counts_prefix = list(range(1, len(sorted_times_prefix) + 1))

        # Extract and sort data for baseline
        problems_baseline = list(data_baseline.keys())
        times_baseline = list(data_baseline.values())
        sorted_data_baseline = sorted(zip(times_baseline, problems_baseline))
        sorted_times_baseline = [item[0] for item in sorted_data_baseline]
        cumulative_counts_baseline = list(range(1, len(sorted_times_baseline) + 1))

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_times_prefix, cumulative_counts_prefix, marker='o', linestyle='-', color='b', label='Prefix Sampling')
        plt.plot(sorted_times_baseline, cumulative_counts_baseline, marker='x', linestyle='--', color='r', label='Baseline')
        plt.xlabel('Time s')
        plt.xscale('log')
        plt.ylabel('Number of Problems Solved')
        plt.title('Problems Solved Over Time')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.savefig("plot.png")

    except Exception as e:
        print(f"An error occurred: {e}")

# Entry point for the script
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <experiment_name>")
    else:
        experiment_name = sys.argv[1]
        plot_problems_solved(experiment_name)