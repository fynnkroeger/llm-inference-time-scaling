from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from shared_utils.code_evaluation.utils import judge_problems, get_task_ids_and_prompt_token_ids_for_non_solved_problems, get_all_task_ids_and_prompts
from shared_utils.code_evaluation.runner import test_results_cache

from os import environ
import random
import math
import time
from mcts.token_ids_prefix_tree import BaseTokenIdsPrefixTree

def run_iterative_baseline(config):
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
    other_time_per_gen = []
    start_time = time.time()
    
    samples = []
    solved_task_ids = {}

    for k in range(0, n, generation_step_size):
        task_ids, prompt_token_ids = get_task_ids_and_prompt_token_ids_for_non_solved_problems(solved_task_ids, problems)
        
        # print(prompt_token_ids)
        
        raw_outputs = llm.generate(prompt_token_ids = prompt_token_ids, sampling_params = sampling_params)
        
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
        
        t_1 = time.time()     
        solved_problems, judged_samples = judge_problems(new_samples, task_ids, start_time = start_time)
        other_time_per_gen.append(time.time() - t_1)
        
        samples += judged_samples
        solved_task_ids = solved_task_ids | solved_problems
        
        time_per_gen.append(time.time() - start_time)
        
    return samples, solved_task_ids, time_per_gen, other_time_per_gen