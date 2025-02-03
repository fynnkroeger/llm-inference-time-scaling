from copy import deepcopy
import json
from math import isnan
from typing import Union

from gguf import Optional
from human_eval.data import write_jsonl # type: ignore
import torch
import os
import warnings
from pathlib import Path
def read_samples(file_path: str) -> list:
    data = []
    base_path = file_path.replace(".jsonl", "").replace(".gz", "")

    hidden_states_file_path = base_path + ".pt"
    if os.path.isfile(hidden_states_file_path):
        print("Loading hidden states from file")
        hidden_states: Optional[dict[str, torch.Tensor]] = torch.load(hidden_states_file_path, weights_only=True)
        print(f"Loaded {len(hidden_states)} hidden states")
    else:
        print("Found no hidden states file!")
        hidden_states = None

    with open(file_path, "r") as f:
        for line in f.readlines():
            new_sample = json.loads(line)
            if "hidden_states" in new_sample:
                if hidden_states is None:
                    raise Exception(f".jsonl sample file contained hidden_states ids but no hidden_states were found in: {hidden_states_file_path}!")
                else:
                    key = new_sample["hidden_states"]
                    # loads via id placeholder saved in the samples
                    
                    hidden_state = hidden_states[key]
                    inf_mask = torch.isinf(hidden_state)
                    hidden_state[inf_mask] = 0.0
                    new_sample["hidden_states"] = hidden_state

            data.append(new_sample)
    return data

def write_samples(file_path: Union[str, Path], results: list[dict]) -> None:
    json_serializable_results = []
    hidden_states = {}
    file_path = str(file_path)
    total_nan_replacements = 0
    
    for i, result in enumerate(results):
        result = deepcopy(result)
        if "hidden_states" in result:
            # Count and replace NaNs with zeros
            nan_mask = torch.isnan(result["hidden_states"])
            num_nans = nan_mask.sum().item()
            inf_mask = torch.isinf(result["hidden_states"])
            if num_nans > 0:
                result["hidden_states"][nan_mask] = 0.0
                result["hidden_states"][inf_mask] = 0.0
                total_nan_replacements += num_nans
            
            hidden_states_id = f"{result['task_id']}/{i}"
            hidden_states[hidden_states_id] = result["hidden_states"]
            result["hidden_states"] = hidden_states_id
            
        json_serializable_results.append(result)
    
    if total_nan_replacements > 0:
        warnings.warn(f"WARNING: Replaced {total_nan_replacements} NaN values with zeros in hidden states (likely due to LLaMA quantization)")
    
    base_path = file_path.replace(".jsonl", "")
    write_jsonl(base_path + ".jsonl", json_serializable_results)
    torch.save(hidden_states, base_path + ".pt")


def get_task_ids_and_prompts_for_non_solved_problems(
    solved_problems: dict[str, bool], problems: dict
) -> tuple[list[str], list[str]]:
    prompts = []
    task_ids = []
    for task_id in problems:
        if task_id not in solved_problems:
            prompt = problems[task_id]["prompt"]
            task_ids.append(task_id)
            prompts.append(prompt)
    return task_ids, prompts

from shared_utils.code_evaluation.runner import evaluate_only_functional_correctness

def judge_problems(outputs: list, task_ids: list[str], extract_function_outputs: bool = False, hash_function_outputs: bool = True) -> tuple[dict[str, bool], list]:
    results = evaluate_only_functional_correctness(outputs, n_workers=64, extract_function_outputs=extract_function_outputs, hash_function_outputs=hash_function_outputs)

    solved_problems = {}
    for result in results:
        if result["passed"]:
            solved_problems[result["task_id"]] = True
            
    return solved_problems, results