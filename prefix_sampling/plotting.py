
from pathlib import Path

import json
import sys
import matplotlib.pyplot as plt

DEBUG = True


experiment_path = Path("/raid/shared/llm-inference-scaling/prefix_sampling_experiments")

# Function to plot the number of problems solved over time
def plot_problems_solved(experiment_name):
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
    if DEBUG:
        plt.savefig("plot_solutions_over_time.png")
    else:
        plt.savefig(ex_path / "plot_solutions_over_time.png")
        
# Function to plot the time per generation
def plot(experiment_name, name):
    # Load JSON data from files
    ex_path = experiment_path / experiment_name
    file_path_prefix = ex_path / (name + "_prefix_sampling.json")
    file_path_baseline = ex_path / (name + "_baseline.json")
    with open(file_path_prefix, 'r') as file:
        data_prefix = json.load(file)

    with open(file_path_baseline, 'r') as file:
        data_baseline = json.load(file)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(data_prefix, marker='o', linestyle='-', color='b', label='Prefix Sampling')
    plt.plot(data_baseline, marker='x', linestyle='--', color='r', label='Baseline')
    plt.xlabel('Generation')
    plt.ylabel(name)
    plt.title(name)
    plt.legend()
    plt.grid(True)

    # Save the plot
    if DEBUG:
        plt.savefig("plot_" + name + ".png")
    else:
        plt.savefig(ex_path / ("plot_" + name + ".png"))
        
# Function to plot the time per generation
def plot_proportional(experiment_name, name):
    # Load JSON data from files
    ex_path = experiment_path / experiment_name
    file_path_prefix = ex_path / (name + "_prefix_sampling.json")
    file_path_baseline = ex_path / (name + "_baseline.json")
    file_path_prefix_nums = ex_path / "num_problems_prefix_sampling.json"
    file_path_baseline_nums = ex_path / "num_problems_baseline.json"
    
    with open(file_path_prefix, 'r') as file:
        data_prefix = json.load(file)

    with open(file_path_baseline, 'r') as file:
        data_baseline = json.load(file)
        
    with open(file_path_prefix_nums, 'r') as file:
        nums_prefix = json.load(file)

    with open(file_path_baseline_nums, 'r') as file:
        nums_baseline = json.load(file)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot([data_prefix[i] / nums_prefix[i] for i in range(len(data_prefix))], marker='o', linestyle='-', color='b', label='Prefix Sampling')
    plt.plot([data_baseline[i] / nums_baseline[i] for i in range(len(data_baseline))], marker='x', linestyle='--', color='r', label='Baseline')
    plt.xlabel('Generation')
    plt.ylabel(name)
    plt.title(name)
    plt.legend()
    plt.grid(True)

    # Save the plot
    if DEBUG:
        plt.savefig("plot_" + name + "_proportional.png")
    else:
        plt.savefig(ex_path / ("plot_" + name + "_proportional.png"))

# Entry point for the script
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <experiment_name>")
    else:
        experiment_name = sys.argv[1]
        plot_problems_solved(experiment_name)
        plot(experiment_name, "other")
        plot(experiment_name, "gen_time")
        plot(experiment_name, "pure_gen_time")
        plot_proportional(experiment_name, "gen_time")
        plot_proportional(experiment_name, "pure_gen_time")