import re
from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from shared_utils.code_evaluation.utils import judge_problems, get_task_ids_and_prompts_for_non_solved_problems
from shared_utils.code_evaluation.runner import test_results_cache, task_id_to_input_output_pairs
from statistics import mean, stdev

import os
from mcts.token_ids_prefix_tree import ExpectedValueSearchTree
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" 

validation_prompt = """
You are an AI specialized in Python syntax validation. Given a Python code snippet, determine if it has valid syntax.

### Response Rules:
- Reply only with **"Yes"** or **"No"**—nothing else.
- Return **"Yes"** if the syntax is valid and could be completed without immediate errors.
- Return **"No"** if the code has **any syntax errors** that make it **immediately invalid** (e.g., missing colons, unclosed parentheses, indentation issues, malformed keywords).
- **Do not** provide explanations, formatting, or any other output beyond "Yes" or "No."

### Examples:

Valid (Returns "Yes")**:
```python
def add_numbers(a, b):
    value = a + b
    retu
```

Valid (Returns "Yes"):
```python
for i in range(10)
```

Invalid (Returns "No"):
```python
def add_numbers(a, b)
     value = (a + b
    retu
```

Now, analyze the following code and respond only with "Yes" or "No":
"""

prompt_ = [
    {"role": "system", "content": "You are an AI specialized in Python syntax validation. You must answer 'Yes' or 'No'."},

    {"role": "user", "content": "Return 'No' if the code has **any syntax errors** that make it **immediately invalid** (e.g., missing colons, unclosed parentheses, indentation issues, malformed keywords). Valid (Returns 'Yes'). Please only answer 'Yes' or 'No'."},

    # Example 1: Valid code
    {"role": "user", "content": "Is this code valid?\n```python\ndef add_numbers(a, b):\n    return a + b\n```"},
    {"role": "assistant", "content": "Yes"},

    # Example 2: Invalid code (fehlendes `:`)
    {"role": "user", "content": "Is this code valid?\n```python\ndef add_numbers(a, b)\n    return a + b\n```"},
    {"role": "assistant", "content": "No"}
]

def generate_vllm_chat(code_snippet):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI specialized in Python syntax validation. You must answer 'Yes' or 'No'.<|eot_id|><|start_header_id|>user<|end_header_id|>

Return 'No' if the code has **any syntax errors** that make it **immediately invalid** (e.g., missing colons, unclosed parentheses, indentation issues, malformed keywords). Valid code returns 'Yes'. Please only answer 'Yes' or 'No'.

### Examples:
#### ✅ Example 1: Valid code
```python
def add_numbers(a, b):
    return a + b
```
Assistant's answer: **Yes**

#### ❌ Example 2: Invalid code (missing `:`)
```python
def add_numbers(a, b)
    return a + b
```
Assistant's answer: **No**<|eot_id|><|start_header_id|>user<|end_header_id|>

Is this code valid?
```python
{code_snippet}
```<|end_header_id|>assistant<|end_header_id|>

"""


problems = read_problems()
n = 256

GENERATION_STEP_SIZE = 1

assert n % GENERATION_STEP_SIZE == 0, "n must be divisble by the GENERATION_STEP_SIZE"

out_path = Path("outputs_tony")
out_path.mkdir(exist_ok=True)

llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct")
# syntax_model = LLM(model="meta-llama/Llama-3.2-1B-Instruct", device="cuda:4")

# Hyperparamter
h_params = {}
h_params["temperature"] = 0.4
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
        syntax_validations_error_tokens = {}
        for task_id, output in zip( task_ids, raw_outputs):
            for i in range(GENERATION_STEP_SIZE):
                completion_output = output.outputs[i]
                
                logprobs = []
                
                validation_sequence = ""
                error_token = 0
                has_error = False
                function_begin = re.sub(r'""".*?"""', '', output.prompt, flags=re.DOTALL).strip()
                code_snippet = function_begin + completion_output.text

                is_valid = llm.generate(
                    generate_vllm_chat(code_snippet),
                    SamplingParams(temperature=0.0),
                    use_tqdm=False
                )[0].outputs[0].text
                
                
                number_of_nos = 0
                
                for logprob in completion_output.logprobs:
                    token_id, info = list(logprob.items())[0] # Each logprob is a dict: {220: Logprob(logprob=0.0, rank=1, decoded_token=' ')}
                    

                    logprobs.append( {
                        "token_id": token_id,
                        "logprob": info.logprob,
                        "rank": info.rank,
                        "decoded_token": info.decoded_token
                    })
                    
                    if not has_error and not (is_valid == "Yes"):
                        number_of_nos += 1
                        validation_sequence += info.decoded_token
                        code_snippet_seq = function_begin + validation_sequence
                        value = llm.generate(generate_vllm_chat(code_snippet_seq),
                                                SamplingParams(temperature=0.0),
                                                use_tqdm=False)[0].outputs[0].text
                        if (value == "No"):
                            syntax_validations_error_tokens[task_id] = error_token
                            has_error = True
                        error_token += 1
                
            
            new_samples.append({
                "task_id": task_id,
                "prompt_token_ids": output.prompt_token_ids,
                "prompt": output.prompt,
                "completion": completion_output.text,
                "cumulative_logprob": completion_output.cumulative_logprob,
                "logprobs": logprobs,
            })
        
        solved_problems, judged_samples = judge_problems(new_samples, task_ids, extract_function_outputs=is_extract_function_outputs)

        for judged_output in judged_samples:
            prompt = judged_output["prompt"]
            raw_logprobs = []
            output_token_ids = []
            decoded_tokens = []
            for x in judged_output["logprobs"]:
                raw_logprobs.append(x["logprob"])
                output_token_ids.append(x["token_id"])
                decoded_tokens.append(x["decoded_token"])
            
            token_count = syntax_validations_error_tokens.get(judged_output["task_id"], len(output_token_ids))
            tree.add_sequence(judged_output["prompt_token_ids"], output_token_ids[:token_count], raw_logprobs[:token_count])
            # tree.add_sequence(judged_output["prompt_token_ids"], output_token_ids, raw_logprobs)

        
        samples += judged_samples
        solved_task_ids = solved_task_ids | solved_problems
        if len(solved_problems) > 0:
            print(f"K: {k} T: {temperature}. Newly solved problems: {len(solved_problems)}. Total solved problems: {len(solved_task_ids)}.")
        
    

    write_jsonl(out_path / f"Suffix_3B_Instruct_llm-verification_temp04.jsonl_results.jsonl", samples)
    test_results_cache.save()