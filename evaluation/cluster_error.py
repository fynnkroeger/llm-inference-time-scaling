import json
from collections import Counter
from typing import List, Dict

def read_jsonl(file_path: str) -> List[Dict]:
    """
    Reads a .jsonl file and returns a list of dictionaries.
    
    Args:
        file_path (str): Path to the .jsonl file.
        
    Returns:
        List[Dict]: List of dictionaries parsed from the .jsonl file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def group_by_task(entries: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Groups entries by task ID.
    
    Args:
        entries (List[Dict]): List of JSON entries.
        
    Returns:
        Dict[str, List[Dict]]: Dictionary where keys are task IDs and values are lists of entries for each task.
    """
    grouped = {}
    for entry in entries:
        task_id = entry.get("task_id")
        if task_id not in grouped:
            grouped[task_id] = []
        grouped[task_id].append(entry)
    return grouped

def group_by_sample(entries: List[Dict], num_samples: int) -> List[List[Dict]]:
    """
    Groups entries by sample, where each sample consists of entries with the same index mod num_samples.
    
    Args:
        entries (List[Dict]): List of JSON entries.
        num_samples (int): The number of entries in each sample.
        
    Returns:
        List[List[Dict]]: A list of samples, where each sample is a list of entries.
    """
    grouped = {}
    for i, entry in enumerate(entries):
        if i % num_samples not in grouped:
            grouped[i % num_samples] = []
        grouped[i % num_samples].append(entry)

    return grouped

def count_errors_per_category(grouped_entries: Dict[str, List[Dict]]) -> Dict[str, Counter]:
    """
    Counts errors for each category (task or sample) and organizes them in a Counter object.
    
    Args:
        grouped_entries (Dict[str, List[Dict]]): Entries grouped by category (task or sample).
        
    Returns:
        Dict[str, Counter]: Dictionary of error counts per category.
    """
    error_counts = {}
    for category_id, entries in grouped_entries.items():
        error_counter = Counter()
        for entry in entries:
            result = entry.get("result", "")
            if result.startswith("failed"):
                error_type = result.split(":")[1].split("-")[0].strip()
                if error_type != "AssertionError":
                    error_counter[error_type] += 1
        error_counts[category_id] = error_counter
    return error_counts


def find_category_with_most_errors(error_counts: Dict[str, Counter]) -> str:
    """
    Finds the category ID with the most total errors.
    
    Args:
        error_counts (Dict[str, Counter]): Dictionary of error counts per category.
        
    Returns:
        str: Category ID with the most errors.
    """
    return max(error_counts, key=lambda category: sum(error_counts[category].values()))

def find_category_with_least_errors(error_counts: Dict[str, Counter]) -> str:
    """
    Finds the category ID with the least total errors.
    
    Args:
        error_counts (Dict[str, Counter]): Dictionary of error counts per category.
        
    Returns:
        str: Category ID with the least errors.
    """
    return min(error_counts, key=lambda category: sum(error_counts[category].values()))

def total_errors_for_category(error_counts: Dict[str, Counter], category_id: str) -> int:
    """
    Calculates the total number of errors for a specific category.
    
    Args:
        error_counts (Dict[str, Counter]): Dictionary of error types and their counts.
        category_id (str): The category ID to check.
        
    Returns:
        int: The total number of errors for the specified category.
    """
    return sum(error_counts[category_id].values())

def get_errors_per_category(base_path: str, num_samples: int, group_by_tasks=True, print_all=False):
    """
    Main function to process JSONL data, count errors, and find the category with the most errors.
    
    Args:
        base_path (str): Path to the .jsonl file to analyze.
    """
    try:
        # Read entries from JSONL file
        entries = read_jsonl(base_path)
        
        category = "task"
        # Group entries by specified category (default: tasks)
        if group_by_tasks:
            grouped_entries = group_by_task(entries)
        else:
            category = "sample"
            grouped_entries = group_by_sample(entries, num_samples)
        
        # Count errors per category
        error_counts = count_errors_per_category(grouped_entries)
        
        # Find the category with the most errors
        most_errors_category = find_category_with_most_errors(error_counts)
        least_errors_category = find_category_with_least_errors(error_counts)
        
        # Print the summary
        print("\n----- ERROR SUMMARY -----\n")
        if print_all:
            print(f"\nAll error counts per {category}:")
            print(json.dumps({category: dict(errors) for category, errors in error_counts.items()}, indent=4))
        else:   
            print(f"{category} with the most errors:", most_errors_category)
            print(f"Error details for the {category}:")
            print(json.dumps(error_counts[most_errors_category], indent=4))
            print("Total Errors", total_errors_for_category(error_counts, most_errors_category))
            print(f"\n{category} with the least errors:", least_errors_category)
            print(f"Error details for the {category}:")
            print(json.dumps(error_counts[least_errors_category], indent=4))
            print("Total Errors", total_errors_for_category(error_counts, least_errors_category))
         
    except FileNotFoundError:
        print(f"Error: File not found at {base_path}.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in {base_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# To run the script you have generate the .jsonl_results.jsonl file
# with an adjusted check_correctness function in human_eval.execution
# Set the raised error in line 62 to:
# except BaseException as e:
#     error_type = type(e).__name__
#     error_message = str(e)
#     result.append(f"failed: {error_type} - {error_message}") 
if __name__ == "__main__":
    # Define the path to your .jsonl_results.jsonl file
    name = "wealthy-pegasus-of-massive-prosperity"
    base_path = f"/raid/shared/llm-inference-scaling/experiments_test/{name}/{name}.jsonl_results.jsonl"
    
    # TODO: Should we read from config?
    num_samples = 3
    
    # If set to true all tasks and their error counts are printed
    # If set to false only the task with the most errors is printed
    print_all = True
    
    # If set to true errors are grouped by task
    # If set to false errors are grouped by sample
    group_by_tasks = False
    
    get_errors_per_category(base_path, num_samples, group_by_tasks, print_all)
