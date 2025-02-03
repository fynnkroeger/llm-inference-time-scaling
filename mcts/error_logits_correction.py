import re
from vllm import LLM, SamplingParams
from human_eval.data import read_problems # type: ignore
from pathlib import Path
from shared_utils.code_evaluation.utils import judge_problems, get_task_ids_and_prompts_for_non_solved_problems, write_samples
from shared_utils.code_evaluation.runner import test_results_cache, task_id_to_input_output_pairs
from statistics import mean, stdev

import os
from mcts.token_ids_prefix_tree import ExpectedValueSearchTree
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

problems = read_problems()
# n = 1024
n = 128

GENERATION_STEP_SIZE = 1

assert n % GENERATION_STEP_SIZE == 0, "n must be divisble by the GENERATION_STEP_SIZE"

out_path = Path("outputs_tony")
out_path.mkdir(exist_ok=True)

llm = LLM(model="meta-llama/Llama-3.1-8B")

# Hyperparamter
h_params = {}
h_params["temperature"] = 0.8
h_params["top_p"] = 0.95
h_params["max_tokens"] = 128
is_extract_function_outputs = False

REMOVE_SOLVED_PROBLEMS = True

    
for temperature in [h_params["temperature"]]:
    samples = []
    solved_task_ids = {}
    

    tree = ExpectedValueSearchTree()
    def logit_processor(prompt, output, logits):
        return tree.adjust_logits_fast(prompt, output, logits)
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=h_params["top_p"],
        max_tokens=h_params["max_tokens"],
        logprobs=0,
        n=GENERATION_STEP_SIZE,
        logits_processors=[logit_processor]
    ) # logprobs includes 1 (decoded token) + $logprobs
    for k in range(0, n, GENERATION_STEP_SIZE):
        task_ids, prompts = get_task_ids_and_prompts_for_non_solved_problems(solved_task_ids if REMOVE_SOLVED_PROBLEMS else {}, problems)
        raw_outputs = llm.generate(prompts, sampling_params)        

        new_samples = []
        
        for task_id, output in zip( task_ids, raw_outputs):
            for i in range(GENERATION_STEP_SIZE):
                completion_output = output.outputs[i]
                
                logprobs = []
    
                for logprob in completion_output.logprobs: # type: ignore
                    token_id, info = list(logprob.items())[0] # Each logprob is a dict: {220: Logprob(logprob=0.0, rank=1, decoded_token=' ')}
                  

                    logprobs.append( {
                        "token_id": token_id,
                        "logprob": info.logprob,
                        "rank": info.rank,
                        "decoded_token": info.decoded_token
                    })
                
                
                new_samples.append({
                    "task_id": task_id,
                    "prompt_token_ids": output.prompt_token_ids,
                    "prompt": output.prompt,
                    "completion": completion_output.text,
                    "cumulative_logprob": completion_output.cumulative_logprob,
                    "logprobs": logprobs,
                    "hidden_states": completion_output.hidden_states
                })
        
        solved_problems, judged_samples = judge_problems(new_samples, task_ids, extract_function_outputs=is_extract_function_outputs)
        early_stopped_error_generations = []
        generations_with_errors = 0
        generations_with_wrong_tests = 0

        for judged_output in judged_samples:
            prompt = judged_output["prompt"]
            raw_logprobs = []
            output_token_ids = []
            decoded_tokens = []
            for x in judged_output["logprobs"]:
                raw_logprobs.append(x["logprob"])
                output_token_ids.append(x["token_id"])
                decoded_tokens.append(x["decoded_token"])
            
            # If error is detected we want to adjust logits more strict
            # Find min line where error occur
            min_line = h_params["max_tokens"]
            min_line_index = 0
            error_message = ""
            has_found_error = False
            has_assertion_error = False
            if not judged_output["passed"]:
                #print(judged_output["function_outputs"])
                for index, func_output in enumerate(judged_output["function_outputs"]):
                    if isinstance(func_output, str) and not func_output.startswith("failed: <class 'AssertionError'>"):
                        generations_with_errors += 1
                        has_found_error = True
                        matches = re.findall(r'line (\d+)', func_output)
                        if matches:
                            line_numbers = [int(line) for line in matches]
                            min_line_number = min(line_numbers)
                            if min_line_number < min_line:
                                min_line = min_line_number
                                min_line_index = index
                    else:
                        has_assertion_error = True
            #import pdb; pdb.set_trace()
            
                
            if min_line != 128:
                error_message = judged_output["function_outputs"][min_line_index]
                prompt_lines = prompt.split("\n")              
                completion_lines = judged_output["completion"].split("\n")
                
                problem = problems[judged_output["task_id"]]
                if is_extract_function_outputs and task_id_to_input_output_pairs is not None:
                    program = prompt + judged_output["completion"] + "\n"
                    check_program = f"""
{program}


function_output = {problem["entry_point"]}(*raw_function_input)
"""
                else:
                    check_program = (
                        prompt + judged_output["completion"] + "\n" +
                        problem["test"] + "\n" +
                        f"check({problem['entry_point']})"
                    )
                total_program_lines = check_program.split("\n")
                
                try:
                    error_line = total_program_lines[min_line - 1]
                except Exception as e:
                    error_line = total_program_lines[-1]
                temp_error_line = error_line
                token_count = 0
                for token in decoded_tokens :
                    if temp_error_line.startswith(token):
                        temp_error_line = temp_error_line[len(token):]
                    else:
                        temp_error_line = error_line
                        
                    token_count += 1
                        
                    if len(temp_error_line) == 0:
                        break
                early_stopped_error_generations.append(token_count / len(raw_logprobs))

                tree.add_sequence(judged_output["prompt_token_ids"], output_token_ids[:token_count], raw_logprobs[:token_count], judged_output["hidden_states"])
            elif has_found_error:
                print(min_line, judged_output["function_outputs"])
                tree.add_sequence(judged_output["prompt_token_ids"], output_token_ids, raw_logprobs, judged_output["hidden_states"]) 
            elif has_assertion_error:
                generations_with_wrong_tests += 1
                tree.add_sequence(judged_output["prompt_token_ids"], output_token_ids, raw_logprobs, judged_output["hidden_states"]) 
            # else:
                # tree.add_sequence(judged_output["prompt_token_ids"], output_token_ids, raw_logprobs, judged_output["hidden_states"]) 

            # tree.add_sequence(judged_output["prompt_token_ids"], output_token_ids, raw_logprobs, hash(judged_output["function_outputs"]))
        
        print(f"Adjusted probabilities for {len(early_stopped_error_generations)} sequences with erros. Average length factor: {mean(early_stopped_error_generations)}")
        print(f"Adjusted for {generations_with_wrong_tests} failed tests")

        
        samples += judged_samples
        solved_task_ids = solved_task_ids | solved_problems
        if len(solved_problems) > 0:
            print(f"K: {k} T: {temperature}. Newly solved problems: {len(solved_problems)}. Total solved problems: {len(solved_task_ids)}. Generations with erros: {generations_with_errors}")
        
    

    write_samples(out_path / f"samples-t{str(temperature)}-with-error-correction-8B.jsonl", samples)
    test_results_cache.save()