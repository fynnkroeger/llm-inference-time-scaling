import json
from collections import Counter
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
from pprint import pprint

def read_jsonl(file_path: Path) -> list[dict]:
    """
    Reads a .jsonl file and returns a list of dictionaries.
    
    Args:
        file_path (Path): Path to the .jsonl file.
        
    Returns:
        list[dict]: List of dictionaries parsed from the .jsonl file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file]
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in {file_path}.")

def group_by_task(entries: list[dict]) -> dict[str, list[dict]]:
    """
    Groups entries by task ID.
    
    Args:
        entries (list[dict]): List of JSON entries.
        
    Returns:
        dict[str, list[dict]]: Dictionary where keys are task IDs and values are lists of entries for each task.
    """
    grouped = {}
    for entry in entries:
        task_id = entry.get("task_id")
        if task_id not in grouped:
            grouped[task_id] = []
        grouped[task_id].append(entry)
    return grouped

def group_by_sample(entries: list[dict], num_samples: int) -> list[list[dict]]:
    """
    Groups entries by sample, where each sample consists of entries with the same index mod num_samples.
    
    Args:
        entries (list[dict]): List of JSON entries.
        num_samples (int): The number of entries in each sample.
        
    Returns:
        list[list[dict]]: A list of samples, where each sample is a list of entries.
    """
    grouped = {}
    for i, entry in enumerate(entries):
        if i % num_samples not in grouped:
            grouped[i % num_samples] = []
        grouped[i % num_samples].append(entry)

    return grouped

def count_errors_per_category(grouped_entries: dict[str, list[dict]]) -> dict[str, Counter]:
    """
    Counts errors for each category (task or sample) and organizes them in a Counter object.
    
    Args:
        grouped_entries (dict[str, list[dict]]): Entries grouped by category (task or sample).
        
    Returns:
        dict[str, Counter]: Dictionary of error counts per category.
    """
    error_counts = {}
    for category_id, entries in grouped_entries.items():
        error_counter = count_errors(entries)
        error_counts[category_id] = error_counter
    return error_counts

def count_errors(entries: list[dict]):
    """
    Counts specific error types from a list of entries.
    
    Args:
        entries (list[dict]): A list of dictionaries, each representing an entry with a "result" field.
        
    Returns:
        Counter: A Counter object containing the frequency of each error type.
    """
    error_counter = Counter()
    for entry in entries:
        result = entry.get("result", "")
        if result.startswith("failed"):
            error_type = result.split(":")[1].split("-")[0].strip()
            if error_type != "AssertionError":
                error_counter[error_type] += 1
    return error_counter

def find_category_with_most_errors(error_counts: dict[str, Counter]) -> str:
    """
    Finds the category ID with the most total errors.
    
    Args:
        error_counts (dict[str, Counter]): Dictionary of error counts per category.
        
    Returns:
        str: Category ID with the most errors.
    """
    return max(error_counts, key=lambda category: sum(error_counts[category].values()))

def find_category_with_least_errors(error_counts: dict[str, Counter]) -> str:
    """
    Finds the category ID with the least total errors.
    
    Args:
        error_counts (dict[str, Counter]): Dictionary of error counts per category.
        
    Returns:
        str: Category ID with the least errors.
    """
    return min(error_counts, key=lambda category: sum(error_counts[category].values()))

def total_errors_for_category(error_counts: dict[str, Counter], category_id: str) -> int:
    """
    Calculates the total number of errors for a specific category.
    
    Args:
        error_counts (dict[str, Counter]): Dictionary of error types and their counts.
        category_id (str): The category ID to check.
        
    Returns:
        int: The total number of errors for the specified category.
    """
    return sum(error_counts[category_id].values())

def mean_errors_per_category(error_counts: dict[str, Counter], category_id: str) -> float:
    """
    Calculates the mean number of errors across all categories.
    
    Args:
        error_counts (dict[str, Counter]): Dictionary of error types and their counts.
        
    Returns:
        float: The mean number of errors per category.
    """
    # Avoid division by zero if no categories are provided
    if not error_counts:
        return 0.0
    
    total_errors = total_errors_for_category(error_counts, category_id)
    num_categories = len(error_counts)
    
    return total_errors / num_categories

def create_errors_per_experiment_plot(experiment_path: Path, entries: list[dict]):
    error_counter = count_errors(entries)
    error_data = pd.DataFrame(list(error_counter.items()), columns=["Error Type", "Count"])

    plt.figure(figsize=(8, 6))
    sns.barplot(data=error_data, x="Error Type", y="Count", hue="Error Type", palette="viridis", legend=False)

    plt.title("Error Counts by Type")
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)  # Optional: Rotation für die x-Achse, falls nötig
    plt.tight_layout()

    output_path = experiment_path / "all_errors.png"
    plt.savefig(output_path, dpi=300)  # dpi=300 für hochauflösende Bilder

def create_donut_plot(experiment_path: Path, error_counter: Counter, category: str, category_id: str):
    labels = list(error_counter.keys())
    sizes = list(error_counter.values())
    
    # Use Seaborn color palette
    colors = sns.color_palette("viridis", len(labels))
    plt.figure(figsize=(6, 6))
    fig, ax = plt.subplots()
    wedges, texts = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
    )
    ax.legend(
        wedges,
        labels,
        title="Fehlerarten",
        loc="center left",
        bbox_to_anchor=(0.925, 0.5),
        fontsize=10,
    )
    plt.gca().add_artist(plt.Circle((0, 0), 0.5, color='white'))
    title = f"Detailed errors for {category}:{category_id}"
    plt.title(title)
    plt.tight_layout()
    
    if experiment_path:
        output_file = experiment_path / f"errors_{category}_{category_id}.png"
        plt.savefig(output_file, dpi=300)
        print(f"Donut plot saved at: {output_file}")

def plot_error_per_category(experiment_path: Path, error_counts: dict[str, Counter], category_name: str):
    
    data = []
    for category, counts in error_counts.items():
        total_errors = sum(counts.values())
        data.append({category_name: category, 'total_errors': total_errors})
    
    # Convert the data to a DataFrame for easier handling by seaborn
    df = pd.DataFrame(data)
    mean_errors = df['total_errors'].mean()
    
    sns.set_palette("viridis", n_colors=len(df))
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_name, y='total_errors', data=df)    
    plt.axhline(y=mean_errors, color='black', linestyle='--', label=f'Mean: {mean_errors:.2f}')
    plt.text(x=len(df) * 1.055, y=mean_errors + 0.1, s='Mean', color='black', va='center', ha='right')
    plt.xlabel(f"{category_name}s")
    plt.ylabel('Number of Errors')
    plt.title(f'Errors per {category_name}')
    plt.gca().set_xticklabels([])

    # Save the plot to a file
    plt.tight_layout()
    if experiment_path:
        output_file = experiment_path / f"{category_name}_error_distribution.png"
        plt.savefig(output_file, dpi=500)
        print(f"Plot per {category_name} plot saved at: {output_file}")

def plot_error_bar_charts_per_category(experiment_path: Path, error_counts: dict[str, Counter], category_name: str, number_bars: int):
    # Flatten the error counts into a list of categories, problems, and their respective error counts
    if len(error_counts) < number_bars:
        raise Exception("Number of bars to big")
    
    sampled_error_counts = dict(random.sample(list(error_counts.items()), number_bars))

    
   # Prepare data for stacked bar chart
    data = []
    for category, counts in sampled_error_counts.items():
        if category_name == "task":
            category = category.split("/")[1]
        for error_type, count in counts.items():
            data.append({category_name: category, 'error_type': error_type, 'error_count': count})

    # Convert the data into a DataFrame for easier handling
    df = pd.DataFrame(data)
    pivot_df = df.pivot_table(index=category_name, columns='error_type', values='error_count', aggfunc='sum', fill_value=0)

    plt.figure(figsize=(12, 8))
    pivot_df.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 8))
    plt.xlabel(f'{category_name}s')
    plt.ylabel('Number of Errors')
    plt.legend(title="Error Types", bbox_to_anchor=(-0.4, 0.5), loc='center left', fontsize=12)
    plt.xticks(rotation=0)
    
    # Save the plot to a file
    plt.tight_layout()
    if experiment_path:
        output_file = experiment_path / f"{category_name}_error_bar_charts_per_{category_name}.png"
        plt.savefig(output_file, dpi=500)
        print(f"Bar charts per {category_name} plot saved at: {output_file}")
    
    
def get_errors_per_category(base_path: str, experiment_name: str, num_samples: int, group_by_tasks=True, print_all=False):
    """
    Main function to process JSONL data, count errors, find the category with the
    most errors and create error count plots.
    
    Args:
        base_path (str): Name of the shared directory with the experiment directory
        experiment_name (str): Name of the experiment to analyze.
        num_samples (int): Number of samples of the experiment
        groub_by_tasks (bool): Flag that indicates wheather to group by tasks or problems
        print_all (bool): Flag that indicates if all counted errors should logged to stdout
    """
    try:
        experiment_path = Path(base_path) / experiment_name
        jsonl_path = experiment_path / f"{experiment_name}.jsonl_results.jsonl"

        # Read entries from JSONL file
        entries = read_jsonl(jsonl_path)
        
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
        print(f"----- EXPERIMENT NAME: {experiment_name} -----")
        if print_all:
            print(f"\nAll error counts per {category}:")
            pprint({category: dict(errors) for category, errors in error_counts.items()})
        else:   
            print(f"{category} with the most errors:", most_errors_category)
            print(f"Error details for the {category}:")
            pprint(error_counts[most_errors_category])
            print("Total Errors", total_errors_for_category(error_counts, most_errors_category))
            print(f"\n{category} with the least errors:", least_errors_category)
            print(f"Error details for the {category}:")
            pprint(error_counts[least_errors_category])
            print("Total Errors", total_errors_for_category(error_counts, least_errors_category))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    
def plot_statistics(base_path: str, experiment_name: str, num_samples: int):
    
    experiment_path = Path(base_path) / experiment_name
    jsonl_path = experiment_path / f"{experiment_name}.jsonl_results.jsonl"

    # Read entries from JSONL file
    entries = read_jsonl(jsonl_path)
    
    create_errors_per_experiment_plot(experiment_path, entries)
    
    task_grouped_entries = group_by_task(entries)
    sample_grouped_entries = group_by_sample(entries, num_samples)
    
    task_error_counts = count_errors_per_category(task_grouped_entries)
    sample_error_counts = count_errors_per_category(sample_grouped_entries)
        
    # Find the category with the most errors
    most_errors_tasks = find_category_with_most_errors(task_error_counts)
    most_errors_samples = find_category_with_least_errors(sample_error_counts)
        
    create_donut_plot(experiment_path, task_error_counts[most_errors_tasks], "task", most_errors_tasks.split("/")[1])
    create_donut_plot(experiment_path, sample_error_counts[most_errors_samples], "sample", most_errors_samples)
    
    plot_error_per_category(experiment_path, task_error_counts, "task")
    plot_error_per_category(experiment_path, sample_error_counts, "sample")
    
    plot_error_bar_charts_per_category(experiment_path, task_error_counts, "task", 20)
    plot_error_bar_charts_per_category(experiment_path, sample_error_counts, "sample", 10)
    
    print(f"\nSuccessfully created plots for {experiment_name}")
    
# To run the script you have generate the .jsonl_results.jsonl file
# with an adjusted check_correctness function in human_eval.execution
# Set the raised error in line 62 to:
# except BaseException as e:
#     error_type = type(e).__name__
#     error_message = str(e)
#     result.append(f"failed: {error_type} - {error_message}") 
if __name__ == "__main__":
    # 
    # Use this if you already generated outputs and evaluated them
    #
    
    base_path = "/raid/shared/llm-inference-scaling/experiments"
    # Define the path to your .jsonl_results.jsonl file
    experiment_name = "hypersonic-micro-gaur-from-tartarus"

    # TODO: Should we read from config?
    num_samples = 256
    
    # If set to true all tasks and their error counts are printed
    # If set to false only the task with the most errors is printed
    print_all = True
    
    # If set to true errors are grouped by task
    # If set to false errors are grouped by sample
    group_by_tasks = False
    
    # get_errors_per_category(base_path, experiment_name, num_samples, group_by_tasks, print_all)
    plot_statistics(base_path, experiment_name, num_samples)
