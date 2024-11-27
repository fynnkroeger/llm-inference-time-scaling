from inference import powers_of_x_up_to, get_prompts, store_results, experiment_setup
import yaml
import os
from vllm import LLM
from vllm.sampling_params import BeamSearchParams
from human_eval.data import write_jsonl
from pathlib import Path
from time import time
from human_eval import evaluation
from os import environ
import coolname
import json
import yaml

def beam_search_entry():
    print("starting beam search")
    try:
        config_path = "./config.yaml"
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        n = 1
        # beam_widths = n * powers_of_x_including_max(4, n)
        beam_widths = [n, 4 * n]
        for beam_width in beam_widths:
            # Set temperature for each run
            config["beam_params"]["beam_width"] = beam_width
            print(f"running with beam width: {beam_width}")
            out_path = run_beam_search(config)
            print(f"Results successfully stored in {out_path}")
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        print(f"Error in YAML file: {e}")

def run_beam_generation(out_file, config):
    sampling_params = config["beam_params"]
    llm_params = config["llm_params"]
    prompts, task_ids = get_prompts()
    
    llm = LLM(**llm_params)
    # if "use_beam_search" in config and config["use_beam_search"] == True:
    params = BeamSearchParams(**sampling_params)
    t0 = time()
    outputs = llm.beam_search(prompts, params)
    elapsed_time = time() - t0
    print(f"generation time: {elapsed_time:.3f}")

    samples = []
    for tid, output in zip(task_ids, outputs):
        for out in output.sequences:
            samples.append(dict(task_id=tid, completion=out.text))
    write_jsonl(out_file, samples)
    
    return elapsed_time

def evaluate_beam_search(out_file, config):
    num_samples = 1024
    evaluation.evaluate_functional_correctness(
            str(out_file), k=powers_of_x_including_max(4, num_samples)
        )

def run_beam_search(config):
    experiments, experiments_file, experiment_name, out_path  = experiment_setup(config)
    out_file = out_path / f"{experiment_name}.jsonl"
    
    # Choose your generation scipt here
    creation_time = run_beam_generation(out_file, config)
    experiments[experiment_name	]["runtime"] = creation_time

    # Only store results when run was successful
    if os.path.exists(out_file):
        store_results(experiments, experiments_file)
    
    if config.get("evaluate", False):
        # Choose your evaluation script here
        evaluate_beam_search(out_file, config)
    
    return out_file

beam_search_entry()