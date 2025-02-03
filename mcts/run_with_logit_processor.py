from vllm import LLM, SamplingParams
from human_eval.data import read_problems # type: ignore
from pathlib import Path
from shared_utils.code_evaluation.utils import judge_problems, get_task_ids_and_prompts_for_non_solved_problems, write_samples
from shared_utils.code_evaluation.runner import test_results_cache

import os
from mcts.value_estimator_prefix_tree import NextTokenValueEstimatorTokenIdsPrefixTree
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # todo do this differently

problems = read_problems()

n = 256

GENERATION_STEP_SIZE = 1
REMOVE_SOLVED_PROBLEMS = False
CHECKPOINTING_STEP = 25
USE_LOGIT_PROCESSOR = False

assert n % GENERATION_STEP_SIZE == 0, "n must be divisble by the GENERATION_STEP_SIZE"

out_path = Path("outputs")
out_path.mkdir(exist_ok=True)

model_id = "meta-llama/Llama-3.2-1B"
llm = LLM(model=model_id)

for temperature in [0.8]:
    samples = []
    solved_task_ids = {}
    

    tree = NextTokenValueEstimatorTokenIdsPrefixTree(model_id)
    def logit_processor(prompt, output, logits):
        return tree._get_advantage_adjusted_logits(prompt, output, logits)
    
    sampling_params = SamplingParams(temperature=temperature, top_p=0.95,max_tokens=128, logprobs=0, n=GENERATION_STEP_SIZE, logits_processors=[logit_processor] if USE_LOGIT_PROCESSOR else None) # logprobs includes 1 (decoded token) + $logprobs
    output_file_name = out_path / f"samples-{model_id.replace("/", "")}-t{str(temperature)}.jsonl"

    for k in range(0, n, GENERATION_STEP_SIZE):
        # Always generate with all to get better value function estimates
        task_ids, prompts = get_task_ids_and_prompts_for_non_solved_problems(solved_task_ids if REMOVE_SOLVED_PROBLEMS else {}, problems)
        raw_outputs = llm.generate(prompts, sampling_params)
        
        

        new_samples = []
        for task_id, output in zip( task_ids, raw_outputs):
            prompt = output.prompt
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
                    "completion": completion_output.text,
                    "cumulative_logprob": completion_output.cumulative_logprob,
                    "logprobs": logprobs,
                    "hidden_states": completion_output.hidden_states
                })
        
        solved_problems, judged_samples = judge_problems(new_samples, task_ids, extract_function_outputs=True)
        
        if USE_LOGIT_PROCESSOR:
            for judged_output in judged_samples:
                raw_logprobs = []
                output_token_ids = []
                for x in judged_output["logprobs"]:
                    raw_logprobs.append(x["logprob"])
                    output_token_ids.append(x["token_id"])
                tree.add_sequence(judged_output["prompt_token_ids"], output_token_ids, raw_logprobs, judged_output["hidden_states"], hash(judged_output["function_outputs"]), judged_output["passed"])
            tree.finished_adding_sequences_watermark()

        samples += judged_samples
        solved_task_ids = solved_task_ids | solved_problems
        if len(solved_problems) > 0:
            print(f"K: {k} T: {temperature}. Newly solved problems: {len(solved_problems)}, Total solved problems: {len(solved_task_ids)}/164")
                
        if k > 0 and k % CHECKPOINTING_STEP == 0:
            write_samples(output_file_name, samples)
    
    write_samples(output_file_name, samples)
    test_results_cache.save()