from pathlib import Path

import json
import matplotlib.pyplot as plt
import sys
import numpy as np
from collections import defaultdict

DEBUG = False

if DEBUG:
    experiment_path = Path("/raid/shared/llm-inference-scaling/prefix_sampling_experiments_test")
else:
    experiment_path = Path("/raid/shared/llm-inference-scaling/prefix_sampling_experiments")

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
    # Helper function for exponential smoothing
    def exponential_smoothing(data, alpha):
        smoothed_data = [data[0]]  # Initialize with the first data point
        for point in data[1:]:
            smoothed_value = alpha * point + (1 - alpha) * smoothed_data[-1]
            smoothed_data.append(smoothed_value)
        return smoothed_data

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
    
# Entry point for the script
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <experiment_name>")
    else:
        experiment_name = sys.argv[1]
        prefix_lengths, completion_lengths = process_jsonl(experiment_name)
        save_prefix_lengths_plots(prefix_lengths, completion_lengths, experiment_name)
