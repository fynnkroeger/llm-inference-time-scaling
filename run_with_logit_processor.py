from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from rl.utils import judge_problems, get_task_ids_and_prompts_for_non_solved_problems
import os
from logit_processors.token_ids_prefix_tree import TokenIdsPrefixTree
os.environ["TOKENIZERS_PARALLELISM"] = "true"

problems = read_problems()

n = 1000

GENERATION_STEP_SIZE = 1

assert n % GENERATION_STEP_SIZE == 0, "n must be divisble by the GENERATION_STEP_SIZE"

out_path = Path("outputs")
out_path.mkdir(exist_ok=True)

llm = LLM(model="meta-llama/Llama-3.2-1B")

    
    
    
for temperature in [0.4, 0.8, 1.2]:
    samples = []
    solved_task_ids = {}
    

    tree = TokenIdsPrefixTree()
    def logit_processor(prompt, output, logits):
        return tree.adjust_logits_fast(prompt, output, logits)
    
    sampling_params = SamplingParams(temperature=temperature, top_p=0.95,max_tokens=128, logprobs=0, n=GENERATION_STEP_SIZE, logits_processors=[logit_processor]) # logprobs includes 1 (decoded token) + $logprobs
    for k in range(0, n, GENERATION_STEP_SIZE):
        task_ids, prompts = get_task_ids_and_prompts_for_non_solved_problems(solved_task_ids, problems)
        raw_outputs = llm.generate(prompts, sampling_params)
        
        new_samples = []
        for task_id, output in zip( task_ids, raw_outputs):
            prompt = output.prompt
            for i in range(GENERATION_STEP_SIZE):
                completion_output = output.outputs[i]
                
                logprobs = []
                raw_output_token_ids = []
                raw_log_probs = []

                for logprob in completion_output.logprobs:
                    token_id, info = list(logprob.items())[0] # Each logprob is a dict: {220: Logprob(logprob=0.0, rank=1, decoded_token=' ')}
                    raw_output_token_ids.append(token_id)
                    raw_log_probs.append(info.logprob)

                    logprobs.append( {
                        "token_id": token_id,
                        "logprob": info.logprob,
                        "rank": info.rank,
                        "decoded_token": info.decoded_token
                    })
                
                tree.add_sequence(output.prompt_token_ids, raw_output_token_ids, raw_log_probs)
                new_samples.append({
                    "task_id": task_id,
                    "completion": completion_output.text,
                    "cumulative_logprob": completion_output.cumulative_logprob,
                    "logprobs": logprobs
                })
        
        solved_problems, judged_samples= judge_problems(new_samples, task_ids)
        if len(solved_problems) > 0:
            print(f"K: {k} T: {temperature}. Newly solved problems: {len(solved_problems)}")
        samples += judged_samples
        solved_task_ids = solved_task_ids | solved_problems

    write_jsonl(out_path / f"samples-t{str(temperature)}-adjusted.jsonl", samples)