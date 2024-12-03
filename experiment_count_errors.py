from inference import experiment_setup, run_generation, evaluate_enhanced, store_results
import yaml
import os

from evaluation.cluster_error import get_errors_per_category, plot_statistics
from inference import BASE_PATH

config_path = "./config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

experiments, experiments_file, experiment_name, out_path  = experiment_setup(config)
out_file = out_path / f"{experiment_name}.jsonl"

# Choose your generation scipt here
out_samples, creation_time = run_generation(out_file, config)
experiments[experiment_name	]["runtime"] = creation_time

# Only store results when run was successful
if os.path.exists(out_file):
    store_results(experiments, experiments_file)

if config.get("evaluate", False):
    # Choose your evaluation script here
    evaluate_enhanced(out_file, out_samples)
print(f"Results successfully stored in {out_path}")

group_by_tasks = True
print_all = False
base_path = BASE_PATH
# get_errors_per_category(base_path, experiment_name, config["sampling"]["n"], group_by_tasks, print_all)
plot_statistics(base_path, experiment_name, config["sampling"]["n"])
