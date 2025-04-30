import json
import sys
from pathlib import Path

DEBUG = True
if DEBUG:
    experiment_path = Path("/raid/shared/llm-inference-scaling/prefix_sampling_experiments_test")
else:
    experiment_path = Path("/raid/shared/llm-inference-scaling/prefix_sampling_experiments")

def process_jsonl(file_path):
    """
    Reads and processes a .jsonl file.
    
    Args:
        file_path (str): The path to the .jsonl file.

    Returns:
        list: A list of dictionaries with the parsed content.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # Parse each line as JSON and append it to the list
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e} | Line: {line}")
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return data
   
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <experiment_name>")
    else:
        exp_name = experiment_name = sys.argv[1]
        
        baseline_file_path = experiment_path / experiment_name / "samples_baseline.jsonl"
        prefix_file_path = experiment_path / experiment_name / "samples_prefix_sampling.jsonl"
        # Process the file
        baseline_parsed_data = process_jsonl(baseline_file_path)
        prefix_parsed_data = process_jsonl(prefix_file_path)

        # Print the parsed data (for demonstration purposes)
        baseline_total = 0
        prefix_total = 0
        for baseline_entry, prefix_entry in zip(baseline_parsed_data, prefix_parsed_data):
            # print(len(baseline_entry["logprobs"]), len(prefix_entry["logprobs"]))
            baseline_total += len(baseline_entry["logprobs"])
            prefix_total += len(prefix_entry["logprobs"])
        print(f"total tokens in completion: baseline {baseline_total}, prefix sampling {prefix_total}")
