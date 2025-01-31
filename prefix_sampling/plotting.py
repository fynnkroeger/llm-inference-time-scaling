from pathlib import Path

import json
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

DEBUG = False

if DEBUG:
    experiment_path = Path("/raid/shared/llm-inference-scaling/prefix_sampling_experiments_test")
else:
    experiment_path = Path("/raid/shared/llm-inference-scaling/prefix_sampling_experiments/multi_2")

def exponential_smoothing(data, alpha):
    smoothed = []
    for i, value in enumerate(data):
        if i == 0:
            smoothed.append(value)  # Initialize with the first value
        else:
            smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
    return smoothed

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

# Function to parse JSONL file and extract prefix lengths and total token counts
def process_jsonl(experiment_name):
    ex_path = experiment_path / experiment_name
    file_path_prefix = ex_path / "samples_prefix_sampling.jsonl"
    
    prefix_lengths = []
    completion_lengths = []

    task_last_step = defaultdict(int)  # Tracks the last generation step for each task_id

    with open(file_path_prefix, 'r') as file:
        current_step_prefix_sum = 0
        current_step_completion_sum = 0
        current_generation_step = 0

        for line in file:
            data = json.loads(line)
            task_id = data.get("task_id")
            prefix_len = data.get("prefix_len", 0)

            # Calculate completion length using the number of entries in logprobs
            logprobs = data.get("logprobs", [])
            completion_len = len(logprobs)

            last_step = task_last_step[task_id]
            if last_step == current_generation_step:
                # A new generation step starts, record cumulative values
                if current_step_completion_sum > 0:
                    prefix_lengths.append(current_step_prefix_sum)
                    completion_lengths.append(current_step_completion_sum)

                # Reset for the new generation step
                current_step_prefix_sum = 0
                current_step_completion_sum = 0
                current_generation_step += 1

            # Update sums for the current step
            current_step_prefix_sum += prefix_len
            current_step_completion_sum += completion_len

            # Update the last generation step for this task_id
            task_last_step[task_id] = current_generation_step

        # Append the last step if not yet added
        if current_step_prefix_sum > 0 and current_step_completion_sum > 0:
            prefix_lengths.append(current_step_prefix_sum)
            completion_lengths.append(current_step_completion_sum)

    return prefix_lengths, completion_lengths

def save_prefix_lengths_plots(prefix_lengths, completion_lengths, experiment_name, alpha=0.2):
    """
    Plots smoothed prefix lengths and percentages using exponential smoothing.
    
    Parameters:
    - prefix_lengths: List of prefix lengths per generation step.
    - completion_lengths: List of completion lengths per generation step.
    - alpha: Smoothing factor for exponential smoothing (0 < alpha <= 1).
    """
    ex_path = experiment_path / experiment_name

    # Calculate percentages
    prefix_percentages = [
        (prefix / total if total > 0 else 0) * 100
        for prefix, total in zip(prefix_lengths, completion_lengths)
    ]

    # Apply exponential smoothing
    smoothed_prefix_lengths = exponential_smoothing(prefix_lengths, alpha)
    smoothed_prefix_percentages = exponential_smoothing(prefix_percentages, alpha)

    # Plot smoothed prefix lengths
    plt.figure(figsize=(8, 6))
    plt.plot(smoothed_prefix_lengths, label='Exponentially Smoothed Prefix Length', color='blue')
    plt.xlabel('Generation Steps')
    plt.ylabel('Prefix Length')
    plt.title('Exponentially Smoothed Prefix Length per Generation Step')
    plt.legend()
    if DEBUG:
        plt.savefig(f"prefix_lengths_exponential_smoothing_{alpha}.png")
    else:
        plt.savefig(ex_path / f"prefix_lengths_exponential_smoothing_{alpha}.png")
    plt.close()

    # Plot smoothed prefix percentages
    plt.figure(figsize=(8, 6))
    plt.plot(smoothed_prefix_percentages, label='Exponentially Smoothed Prefix Percentage', color='orange')
    plt.xlabel('Generation Steps')
    plt.ylabel('Percentage (%)')
    plt.title('Exponentially Smoothed Prefix Length as Percentage of Total Tokens')
    plt.legend()
    if DEBUG:
        plt.savefig(f"prefix_percentages_exponential_smoothing_{alpha}.png")
    else:
        plt.savefig(ex_path / f"prefix_percentages_exponential_smoothing_{alpha}.png")
    plt.close()

def plot(experiment_name, name, alpha=0.2):
    ex_path = experiment_path / experiment_name
    file_path_prefix = ex_path / (name + "_prefix_sampling.json")
    file_path_baseline = ex_path / (name + "_baseline.json")

    with open(file_path_prefix, 'r') as file:
        data_prefix = json.load(file)

    with open(file_path_baseline, 'r') as file:
        data_baseline = json.load(file)

    smoothed_prefix = exponential_smoothing(data_prefix, alpha)
    smoothed_baseline = exponential_smoothing(data_baseline, alpha)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_prefix, marker='o', linestyle='-', color='b', label='Prefix Sampling')
    plt.plot(smoothed_baseline, marker='x', linestyle='--', color='r', label='Baseline')
    plt.xlabel('Generation')
    plt.ylabel(name)
    plt.title(name)
    plt.legend()
    plt.grid(True)

    if DEBUG:
        plt.savefig(f"plot_{name}_{alpha}.png")
    else:
        plt.savefig(ex_path / f"plot_{name}_{alpha}.png")

def plot_proportional(experiment_name, name, alpha=0.1):
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

    proportional_prefix = [data_prefix[i] / nums_prefix[i] for i in range(len(data_prefix))]
    proportional_baseline = [data_baseline[i] / nums_baseline[i] for i in range(len(data_baseline))]

    smoothed_prefix = exponential_smoothing(proportional_prefix, alpha)
    smoothed_baseline = exponential_smoothing(proportional_baseline, alpha)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_prefix, marker='o', linestyle='-', color='b', label='Prefix Sampling')
    plt.plot(smoothed_baseline, marker='x', linestyle='--', color='r', label='Baseline')
    plt.xlabel('Generation')
    plt.ylabel(name)
    plt.title(name)
    plt.legend()
    plt.grid(True)

    if DEBUG:
        plt.savefig(f"plot_{name}_proportional_{alpha}.png")
    else:
        plt.savefig(ex_path / f"plot_{name}_proportional_{alpha}.png")

# Entry point for the script
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <experiment_name>")
    else:
        experiment_name = sys.argv[1]
        alphas = [0.05, 0.1, 1]
        plot_problems_solved(experiment_name)
        prefix_lengths, completion_lengths = process_jsonl(experiment_name)
        for alpha in alphas:
            plot(experiment_name, "other", alpha)
            plot(experiment_name, "gen_time", alpha)
            plot(experiment_name, "pure_gen_time", alpha)
            plot_proportional(experiment_name, "gen_time", alpha)
            plot_proportional(experiment_name, "pure_gen_time", alpha)
            save_prefix_lengths_plots(prefix_lengths, completion_lengths, experiment_name, alpha)