from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from shared_utils.code_evaluation.utils import judge_problems, get_task_ids_and_prompt_token_ids_for_non_solved_problems, get_all_task_ids_and_prompts
from shared_utils.code_evaluation.runner import test_results_cache
from shared_utils.naming_utils import generate_unique_name
from shared_utils.iterative_baseline import run_iterative_baseline

from os import environ
from bisect import bisect_left
import random
import math
import time
import coolname
import json
from mcts.token_ids_prefix_tree import PrefixTreeCumulativeProbabilities

experiment_path = Path("/raid/shared/llm-inference-scaling/prefix_sampling_experiments")

environ["CUDA_VISIBLE_DEVICES"] = "0"  # todo do this differently
environ["TOKENIZERS_PARALLELISM"] = "true" 

def save_to_tree(judged_samples, tree):
    for judged_output in judged_samples:
        raw_logprobs = []
        output_token_ids = []
        for x in judged_output["logprobs"]:
            raw_logprobs.append(x["logprob"])
            output_token_ids.append(x["token_id"])

        tree.add_sequence(judged_output["prompt_token_ids"], output_token_ids, raw_logprobs, hash(judged_output["function_outputs"]), judged_output["task_id"])

def generate_first_iteration(problems: dict, sampling_params, llm, generation_step_size, start_time):
    task_ids, prompts = get_all_task_ids_and_prompts(problems)
    raw_outputs = llm.generate(prompts, sampling_params)
    
    new_samples = []
    for task_id, output in zip(task_ids, raw_outputs):
        prompt = output.prompt
        for i in range(generation_step_size):
            completion_output = output.outputs[i]
            
            logprobs = []

            for logprob in completion_output.logprobs:
                token_id, info = list(logprob.items())[0] # Each logprob is a dict: {220: Logprob(logprob=0.0, rank=1, decoded_token=' ')}
                logprobs.append( {
                    "token_id": token_id,
                    "logprob": info.logprob
                })
            new_samples.append({
                "task_id": task_id,
                "prompt_token_ids": output.prompt_token_ids,
                "completion": completion_output.text,
                "logprobs": logprobs
            })
    return judge_problems(new_samples, task_ids, start_time = start_time)

def run_prefix_experiment(config):
    generation_step_size = config["generation_step_size"]
    temperature = config["temperature"]
    top_p = config["top_p"]
    max_tokens = config["max_tokens"]
    n = config["n"]
    
    assert n % generation_step_size == 0, "n must be divisble by the generation_step_size"

    problems = read_problems()
    llm = LLM(model=config["model"], enable_prefix_caching=False)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, logprobs=0, n=generation_step_size) # logprobs includes 1 (decoded token) + $logprobs
    
    task_ids, prompts = get_all_task_ids_and_prompts(problems)
    prompt_token_ids = tokenizer(prompts)['input_ids']
    problems = dict(zip(task_ids, prompt_token_ids))
    
    time_per_gen = []
    pure_gen_time = []
    other_time_per_gen = []
    num_problems = []
    start_time = time.time()
    
    samples = []
    solved_task_ids = {}
    tree = PrefixTreeCumulativeProbabilities()

    for k in range(0, n, generation_step_size):
        t_2 = time.time()
        task_ids, prompt_token_ids = get_task_ids_and_prompt_token_ids_for_non_solved_problems(solved_task_ids, problems)
        
        new_prompts = []
        prefixes = []
        for prompt_token_id_list in prompt_token_ids:
            prompt_token_id = tuple(prompt_token_id_list)
            prefix = []
            if not prompt_token_id in tree.prompt_root:
                new_prompts = prompt_token_ids
                prefixes = [[] for i in range(len(prompt_token_ids))]
                break
            node = tree.prompt_root[prompt_token_id]
            while True:
                if not node["children_token_ids"]:
                    # There are no saved child tokens, so the LLM has to generate from here
                    break
                r = random.uniform(0, 1)
                if r > node["cumulative_probs"][-1]:
                    break
                index = bisect_left(node["cumulative_probs"], r)
                child_key = node["child_keys"][index]
                child_node = node["children_token_ids"][child_key]
                assert child_node["token_id"] is not None
                # is None for prompt_root_node
                prefix.append(child_node["token_id"])
                node = child_node
                
            new_prompts.append(prompt_token_id_list + prefix)
            prefixes.append(prefix)
            
        
        # TODO: max_tokens should be different, because the prompt now contains part of the output
        t_3 = time.time()
        raw_outputs = llm.generate(prompt_token_ids = new_prompts, sampling_params = sampling_params)
        pure_gen_time.append(time.time() - t_3)
                
        new_samples = []
        for task_id, prompt_token_id_list, output, prefix in zip(task_ids, prompt_token_ids, raw_outputs, prefixes):
            prompt_token_id = tuple(prompt_token_id_list)
            # Add prefix to every output completion
            prefix_logprobs = tree.get_prefix_logprobs(prompt_token_id, prefix)
            for completion_output in output.outputs:
                logprobs = prefix_logprobs

                for logprob in completion_output.logprobs:
                    token_id, info = list(logprob.items())[0] # Each logprob is a dict: {220: Logprob(logprob=0.0, rank=1, decoded_token=' ')}
                    logprobs.append( {
                        "token_id": token_id,
                        "logprob": info.logprob
                    })
                new_samples.append({
                    "task_id": task_id,
                    "prompt_token_ids": prompt_token_id,
                    # Use the origninal promt_token_id instread of the one that was acctually generated on
                    "completion": tokenizer.decode(prefix) + completion_output.text,
                    # Add Prefix string to completion
                    "logprobs": logprobs,
                    "prefix": tokenizer.decode(prefix),
                    "prefix_len": len(prefix)
                })
        
        time_per_gen.append(time.time() - t_2) 
        
        t_1 = time.time()
        solved_problems, judged_samples = judge_problems(new_samples, task_ids, start_time = start_time)
        other_time_per_gen.append(time.time() - t_1)
        
        save_to_tree(judged_samples, tree)
        
        num_problems.append(len(task_ids))
        
        samples += judged_samples
        solved_task_ids = solved_task_ids | solved_problems
        
    return samples, solved_task_ids, time_per_gen, other_time_per_gen, pure_gen_time, num_problems
    
if __name__ == "__main__":
    config = dict(
        generation_step_size = 1,
        temperature = 0.6,
        top_p = 0.95,
        max_tokens = 128,
        n = 8,
        model = "meta-llama/Llama-3.1-8B"
    )
    exp_name = generate_unique_name(experiment_path)
    
    start_round_time = time.time()
    prefix_samples, prefix_solved_task_ids, prefix_time_per_gen, prefix_other, prefix_pure_gen_time, prefix_nums = run_prefix_experiment(config)
    prefix_time = time.time() - start_round_time
    print(f"Prefix samplig: time: {prefix_time}, solved: {len(prefix_solved_task_ids)}")
    
    start_round_time = time.time()
    base_samples, base_solved_task_ids, base_time_per_gen, base_other, base_pure_gen_time, base_nums = run_iterative_baseline(config)
    base_time = time.time() - start_round_time
    print(f"Baseline: time: {base_time}, solved: {len(base_solved_task_ids)}")
    
    output_path = experiment_path / exp_name
    output_path.mkdir(parents=True)
    
    write_jsonl(output_path / f"samples_prefix_sampling.jsonl", prefix_samples)
    with open(output_path / "times_prefix_sampling.json", "w") as f:
        json.dump(prefix_solved_task_ids, f, indent=4)
    with open(output_path / "gen_time_prefix_sampling.json", "w") as f:
        json.dump(prefix_time_per_gen, f, indent=4)
    with open(output_path / "other_prefix_sampling.json", "w") as f:
        json.dump(prefix_other, f, indent=4)
    with open(output_path / "pure_gen_time_prefix_sampling.json", "w") as f:
        json.dump(prefix_pure_gen_time, f, indent=4)
    with open(output_path / "num_problems_prefix_sampling.json", "w") as f:
        json.dump(prefix_nums, f, indent=4)
        
    write_jsonl(output_path / f"samples_baseline.jsonl", base_samples)
    with open(output_path / "times_baseline.json", "w") as f:
        json.dump(base_solved_task_ids, f, indent=4)
    with open(output_path / "gen_time_baseline.json", "w") as f:
        json.dump(base_time_per_gen, f, indent=4)
    with open(output_path / "other_baseline.json", "w") as f:
        json.dump(base_other, f, indent=4)
    with open(output_path / "pure_gen_time_baseline.json", "w") as f:
        json.dump(base_pure_gen_time, f, indent=4)
    with open(output_path / "num_problems_baseline.json", "w") as f:
        json.dump(base_nums, f, indent=4)
        
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"exp_name: {exp_name}")
    