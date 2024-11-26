import os
from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from time import time
from human_eval import evaluation
from os import environ
import coolname
import json
import yaml

BASE_PATH = "/raid/shared/llm-inference-scaling/experiments"

def get_prompts():
    problems = read_problems()
    prompts = [problem["prompt"] for problem in problems.values()]
    task_ids = list(problems.keys())
    
    return prompts, task_ids

def run_generation(out_file, config):
    sampling_params = config["sampling"]
    llm_params = config["llm_params"]
    prompts, task_ids = get_prompts()
    
    llm = LLM(**llm_params)
    t0 = time()
    outputs = llm.generate(prompts, SamplingParams(**sampling_params))
    elapsed_time = time() - t0
    print(f"generation time: {elapsed_time:.3f}")

    samples = []
    for tid, output in zip(task_ids, outputs):
        for out in output.outputs:
            samples.append(dict(task_id=tid, completion=out.text))
    write_jsonl(out_file, samples)
    
    return elapsed_time

def create_experiment_dir(base_path=BASE_PATH):
    base_path = Path(base_path)
    experiment_name = '-'.join(coolname.generate())
    experiment_path = base_path / experiment_name
    experiment_path.mkdir(exist_ok=True, parents=True)
    
    return experiment_name, experiment_path

def store_metadata(name, config, base_path=BASE_PATH):
    experiments_file = Path(base_path) / "experiments.json"
    if experiments_file.exists():
        with open(experiments_file, "r") as f:
            experiments = json.load(f)
    else:
        experiments = {}
    experiments[name] = config
    
    return experiments, experiments_file
    
def experiment_setup(config):
    # Choose GPU
    environ["CUDA_VISIBLE_DEVICES"] = "5"  # todo do this differently
    environ["TOKENIZERS_PARALLELISM"] = "true"
    experiment_name, out_path = create_experiment_dir()
    experiments, experiments_file = store_metadata(experiment_name, config)
    return experiments, experiments_file, experiment_name, out_path
    
def store_results(experiments, experiments_file):
    with open(experiments_file, "w") as f:
        json.dump(experiments, f, indent=4)

def evaluate(out_file, config):
    num_samples = config["sampling"]["n"]
    evaluation.evaluate_functional_correctness(
            str(out_file), k=powers_of_x_up_to(4, num_samples)
        )

def powers_of_x_up_to(x:int, max_value: int) -> list[int]:
    powers = []
    value = 1
    while value <= max_value:
        powers.append(value)
        value *= x
    return powers

def run_experiment(config):   
    # Create experiment dir and store metadata
    experiments, experiments_file, experiment_name, out_path  = experiment_setup(config)
    out_file = out_path / f"{experiment_name}.jsonl"
    
    # Choose your generation scipt here
    creation_time = run_generation(out_file, config)
    experiments[experiment_name	]["runtime"] = creation_time

    # Only store results when run was successful
    if os.path.exists(out_file):
        store_results(experiments, experiments_file)
    
    if config.get("evaluate", False):
        # Choose your evaluation script here
        evaluate(out_file, config)
    
    return out_file


if __name__ == "__main__":
    try:
        config_path = "./config.yaml"
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            
        out_path = run_experiment(config)
        print(f"Results successfully stored in {out_path}")
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        print(f"Error in YAML file: {e}")
    
    
