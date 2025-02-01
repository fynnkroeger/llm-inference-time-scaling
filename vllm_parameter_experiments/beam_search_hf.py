from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from time import time, sleep
from os import environ
import json
import torch
from collections import defaultdict

from vllm_parameter_experiments.run_eval import evaluate_and_save_results, calc_pass_at_k_from_results
from vllm_parameter_experiments.inference import run_experiment, plots_path
import matplotlib.pyplot as plt

experiment_path = Path("/raid/shared/llm-inference-scaling/vllm_parameter_experiments")
output_path = experiment_path / "outputs"
output_path.mkdir(exist_ok=True, parents=True)
experiments_file = experiment_path / "_experiments.json"
plots_path = experiment_path / "plots"
plots_path.mkdir(exist_ok=True, parents=True)


def run_hf(out_file, sampling_params, llm_params, batch_size=16):
    """
    Run HF generation in batches.

    Parameters:
      out_file (str or Path): File to write the generated completions.
      sampling_params (dict): Parameters to pass to `model.generate`.
      llm_params (dict): Parameters for model loading (e.g. "model_name").
      batch_size (int): Number of prompts to process per generation batch.
    """
    problems = read_problems()
    prompts = [problem["prompt"] for problem in problems.values()]
    task_ids = list(problems.keys())

    tokenizer = AutoTokenizer.from_pretrained(llm_params["model_name"], padding_side="left")
    # Make sure the pad token is defined
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        llm_params["model_name"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto"
    )
    samples = []
    t0 = time()

    # We will store the outputs from each batch here.
    total_outputs = []

    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        # Tokenize the batch of prompts
        tokenized = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        input_ids = tokenized.input_ids.to("cuda")
        attention_mask = tokenized.attention_mask.to("cuda")

        with torch.no_grad():
            batch_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                **sampling_params,
                top_k=tokenizer.vocab_size  # disable top_k as is not enabled in vllm
            )
        # Decode the outputs of the batch
        outputs = tokenizer.batch_decode(batch_output, skip_special_tokens=True)
        total_outputs.extend(outputs)

    generation_time = time() - t0

    # Each prompt can generate multiple completions, e.g. when using beam search or repeated sampling.
    # We assume that sampling_params includes "num_return_sequences" if multiple completions are desired.
    num_return_sequences = sampling_params.get("num_return_sequences", 1)

    # Group the outputs with their corresponding task_ids.
    # (Assumes that the outputs are in the same order as the prompts.)
    for tid, idx in zip(task_ids, range(0, len(total_outputs), num_return_sequences)):
        completions = total_outputs[idx: idx + num_return_sequences]
        for out in completions:
            samples.append(dict(task_id=tid, completion=out))

    if samples:
        write_jsonl(out_file, samples)
    return generation_time


# =============================================================================
# The rest of your experiment code remains largely the same.
# For example:

output_files = {}
configs = []

for width in [6, 4, 2]:  # 16 does not work
    print(environ["CUDA_VISIBLE_DEVICES"])
    for early_stopping in [True, False]:
        for repetition_penalty in [1.0]:  # [1.0, 1.1, 1.2]:
            # normal beam search
            configs.append(dict(max_new_tokens=128,
                                num_beams=width, num_return_sequences=width,
                                repetition_penalty=repetition_penalty,
                                do_sample=False, early_stopping=early_stopping))

            # sampling beam search
            for temperature in [0.6, 1.0]:
                configs.append(dict(max_new_tokens=128,
                                    num_beams=width, num_return_sequences=width,
                                    repetition_penalty=repetition_penalty, temperature=temperature,
                                    do_sample=True, early_stopping=early_stopping))

            # diverse beam search
            divisors = [n for n in range(2, width + 1) if width % n == 0]
            for num_beam_groups in divisors:
                for diversity_penalty in [0.5, 1.0, 1.5]:
                    configs.append(dict(max_new_tokens=128,
                                        num_beams=width, num_return_sequences=width,
                                        repetition_penalty=repetition_penalty,
                                        num_beam_groups=num_beam_groups,
                                        diversity_penalty=diversity_penalty, do_sample=False,
                                        early_stopping=early_stopping))

# Additional configurations
for n in [1, 2, 4, 6]:
    configs.append(dict(max_new_tokens=128, do_sample=True, temperature=0.7, num_return_sequences=n))

devices = "4,5,6,7".split(",")

models = ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]
for model in models:
    # vllm runs
    for n in [1, 2, 4, 8, 16, 32, 64]:
        out_file = run_experiment(sampling_params=dict(temperature=0.7, n=n, max_tokens=128),
                                  llm_params=dict(model=model, gpu_memory_utilization=0.75))
        output_files[out_file] = dict(temperature=0.7, n=n, model_name=model)

    # HF runs (now using the batch-enabled run_hf)
    for sampling_params in configs:
        n = 1
        while n <= len(devices):
            environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:n])
            print("Using GPUs:", environ["CUDA_VISIBLE_DEVICES"])
            try:
                # Pass the generation_function=run_hf which now supports batching.
                out_file = run_experiment(
                    sampling_params,
                    llm_params=dict(model_name=model),
                    force_generation=True,
                    generation_function=run_hf
                )
                break
            except RuntimeError:
                n *= 2
        output_files[out_file] = dict(**sampling_params, model_name=model)
        print()

# (Rest of the code remains unchanged, including evaluation and plotting.)
for k, v in output_files.items():
    print(k, v)
result_files = []
for out_file in output_files:
    result_files.append(evaluate_and_save_results(out_file))
sleep(1)
with open(experiments_file, "r") as f:
    experiments = json.load(f)

for model in models:
    times = []  # To store time_taken
    pass_ks = []  # To store pass@k values
    ks = []

    times_rep = []
    pass_ks_rep = []
    ks_rep = []

    vllm_pass = []
    vllm_times = []
    vllm_ks = []

    best_scores = defaultdict(int)
    best_configs = defaultdict(list)
    for (out_file, config), result_file in zip(output_files.items(), result_files):
        if config["model_name"] != model:
            continue
        if "n" in config:
            pass_at_k = calc_pass_at_k_from_results(result_file, [config["n"]])
            time_taken = experiments[str(out_file)]["generation_time"]
            pass_k_value = list(pass_at_k.values())[0]
            vllm_ks.append(config["n"])
            vllm_times.append(time_taken)
            vllm_pass.append(pass_k_value)
        elif "num_beams" in config:
            k = config["num_beams"]
            pass_at_k = calc_pass_at_k_from_results(result_file, [k])
            time_taken = experiments[str(out_file)]["generation_time"]
            pass_k_value = list(pass_at_k.values())[0]
            ks.append(k)
            times.append(time_taken)
            pass_ks.append(pass_k_value)
            if best_scores[k] <= pass_k_value:
                best_scores[k] = pass_k_value
                best_configs[k].append(config)
        else:
            pass_at_k = calc_pass_at_k_from_results(result_file, [config["num_return_sequences"]])
            time_taken = experiments[str(out_file)]["generation_time"]
            pass_k_value = list(pass_at_k.values())[0]
            ks_rep.append(config["num_return_sequences"])
            times_rep.append(time_taken)
            pass_ks_rep.append(pass_k_value)

    print(model)
    print(best_scores)
    for v in best_configs.values():
        for c in v:
            print(c)
        print()

    model_name = model.split("/")[-1]
    plt.figure()
    plt.scatter(times, pass_ks, marker="+", label="HF beam search")
    plt.xlabel("Time Taken (H100-sec)")
    plt.ylabel("pass@k")
    plt.title("Scatter Plot of pass@k vs Time Taken")
    plt.plot(vllm_times, vllm_pass, label="vLLM repeated sampling")
    plt.plot(times_rep, pass_ks_rep, label="HF repeated sampling")
    plt.legend()
    plt.xscale("log")
    plt.savefig(f"out_{model_name}.png")

    plt.figure()
    plt.plot(vllm_ks, vllm_pass, label="vLLM repeated sampling")
    plt.plot(ks_rep, pass_ks_rep, label="HF repeated sampling")
    plt.scatter(ks, pass_ks, marker="+", label="HF beam search")
    plt.legend()
    plt.xscale("log")
    plt.savefig(f"out2_{model_name}.png")
