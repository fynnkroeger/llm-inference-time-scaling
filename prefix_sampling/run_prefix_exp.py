from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from shared_utils.code_evaluation.utils import judge_problems, get_task_ids_and_prompts_for_non_solved_problems, get_all_task_ids_and_prompts
from shared_utils.code_evaluation.runner import test_results_cache

from os import environ
import random
import math
from mcts.token_ids_prefix_tree import BaseTokenIdsPrefixTree

experiment_path = Path("/raid/shared/llm-inference-scaling/prefix_sampling_experiments")
output_path = experiment_path / "outputs"
output_path.mkdir(exist_ok=True, parents=True)
experiments_file = experiment_path / "_experiments.json"
plots_path = experiment_path / "plots"
plots_path.mkdir(exist_ok=True, parents=True)

environ["CUDA_VISIBLE_DEVICES"] = "2"  # todo do this differently
environ["TOKENIZERS_PARALLELISM"] = "true"
    
n = 16
GENERATION_STEP_SIZE = 1

def judge_problems_and_save_to_tree(new_samples, task_ids, tree):
    solved_problems, judged_samples = judge_problems(new_samples, task_ids, extract_function_outputs=True)
    for judged_output in judged_samples:
        raw_logprobs = []
        output_token_ids = []
        for x in judged_output["logprobs"]:
            raw_logprobs.append(x["logprob"])
            output_token_ids.append(x["token_id"])

        tree.add_sequence(judged_output["prompt_token_ids"], output_token_ids, raw_logprobs, hash(judged_output["function_outputs"]), judged_output["task_id"])
    return judged_samples, solved_problems    

def generate_first_iteration(problems: dict, sampling_params, temperature, tree, llm):
    task_ids, prompts = get_all_task_ids_and_prompts(problems)
    raw_outputs = llm.generate(prompts, sampling_params)
    
    new_samples = []
    for task_id, output in zip(task_ids, raw_outputs):
        prompt = output.prompt
        for i in range(GENERATION_STEP_SIZE):
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
    return judge_problems_and_save_to_tree(new_samples, task_ids, tree)

def run_prefix_experiment():
    assert n % GENERATION_STEP_SIZE == 0, "n must be divisble by the GENERATION_STEP_SIZE"

    problems = read_problems()
    llm = LLM(model="meta-llama/Llama-3.2-1B")
        
    for temperature in [1.2]:
        tree = BaseTokenIdsPrefixTree()
        sampling_params = SamplingParams(temperature=temperature, top_p=0.95,max_tokens=128, logprobs=0, n=GENERATION_STEP_SIZE) # logprobs includes 1 (decoded token) + $logprobs
        
        samples, solved_task_ids = generate_first_iteration(problems, sampling_params, temperature, tree, llm)

        for k in range(GENERATION_STEP_SIZE, n, GENERATION_STEP_SIZE):
            task_ids, prompts = get_task_ids_and_prompts_for_non_solved_problems(solved_task_ids, problems)
            prompt_token_ids = tree.get_prompt_token_ids_from_task_ids(task_ids)
            assert not prompt_token_ids == [], "Promts to token_ids should have been converted by previous generation."
            new_prompts = []
            prefixes = []
            # TODO: sample prefix, append to prompt, save to append to output
            for prompt_token_id in prompt_token_ids:
                prefix = []
                node = tree.prompt_root[prompt_token_id]
                while True:
                    if not node["children_token_ids"]:
                        # There are no saved child tokens, so the LLM has to generate from here
                        break
                    r = random.uniform(0, 1)
                    found = False
                    for child_key in node["children_token_ids"]:
                        child_node = node["children_token_ids"][child_key]
                        # TODO: This is not how the llm samples
                        r -= math.exp(child_node["node_log_prob"])
                        if r <= 0:
                            found = True
                            if node["token_id"] is not None:
                                # is None for prompt_root_node
                                prefix.append(node["token_id"])
                            node = child_node
                            break
                    if not found:
                        break
                    
                new_prompts.append(list(prompt_token_id) + prefix)
                prefixes.append(prefix)
            
            raw_outputs = llm.generate(prompt_token_ids = new_prompts, sampling_params = sampling_params)
            
            new_samples = []
            for task_id, prompt_token_id, output, prefix in zip(task_ids, prompt_token_ids, raw_outputs, prefixes):
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
                        "completion": completion_output.text,
                        "logprobs": logprobs
                    })
            
            judged_samples, solved_problems = judge_problems_and_save_to_tree(new_samples, task_ids, tree)
            samples += judged_samples
            solved_task_ids = solved_task_ids | solved_problems
        

        write_jsonl(output_path / f"samples-t{temperature}-n{n}-g{GENERATION_STEP_SIZE}.jsonl", samples)
    
if __name__ == "__main__":
    run_prefix_experiment()