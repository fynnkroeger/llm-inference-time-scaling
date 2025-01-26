from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from time import time
from os import environ
import uuid
import json
import torch
from time import sleep
from vllm_parameter_experiments.run_eval import evaluate_and_save_results, calc_pass_at_k_from_results
from vllm_parameter_experiments.inference import run_experiment, plots_path

experiment_path = Path("/raid/shared/llm-inference-scaling/vllm_parameter_experiments")
output_path = experiment_path / "outputs"
output_path.mkdir(exist_ok=True, parents=True)
experiments_file = experiment_path / "_experiments.json"
plots_path = experiment_path / "plots"
plots_path.mkdir(exist_ok=True, parents=True)


def run_hf(out_file, sampling_params, llm_params):
    problems = read_problems()
    prompts = [problem["prompt"] for problem in problems.values()]
    task_ids = list(problems.keys())

    tokenizer = AutoTokenizer.from_pretrained(llm_params["model_name"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(llm_params["model_name"],
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map="auto")
    samples = []
    t0 = time()

    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    # Generate text using beam search
    with torch.no_grad():
        beam_output = model.generate(
            tokenized.input_ids.to("cuda"),
            attention_mask=tokenized.attention_mask.to("cuda"),
            pad_token_id=tokenizer.eos_token_id,
            **sampling_params,
        )
    outputs = tokenizer.batch_decode(beam_output, skip_special_tokens=True)
    generation_time = time() - t0
    iterators = [iter(outputs)] * (len(outputs) // len(task_ids))
    for tid, output in zip(task_ids, zip(*iterators, strict=True)):
        for out in output:
            samples.append(dict(task_id=tid, completion=out))

    if samples:
        write_jsonl(out_file, samples)
    return generation_time


if __name__ == "__main__":
    output_files = {}
    configs = []

    for width in [2, 4, 6]:  # 16 does not work
        print(environ["CUDA_VISIBLE_DEVICES"])
        for early_stopping in [True, False]:
            for repetition_penalty in [1.0, 1.1, 1.2]:
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
                for num_beam_groups in [2, width]:
                    for diversity_penalty in [1.0]:
                        configs.append(dict(max_new_tokens=128,
                                            num_beams=width, num_return_sequences=width,
                                            repetition_penalty=repetition_penalty,
                                            num_beam_groups=num_beam_groups,
                                            diversity_penalty=diversity_penalty, do_sample=False,
                                            early_stopping=early_stopping))

# todo higher number with more gpus
for n in [1, 2, 4, 6]:
    configs.append(dict(max_new_tokens=128, do_sample=True, temperature=0.7, num_return_sequences=n))

# todo more models, try except cuda out of memory
models = ["meta-llama/Llama-3.2-1B"]
for model in models:
    for n in [1, 2, 4, 8, 16, 32, 64]:
        out_file = run_experiment(sampling_params=dict(temperature=0.7, n=n, max_tokens=128),
                                  llm_params=dict(model=model, gpu_memory_utilization=0.75))
        output_files[out_file] = dict(temperature=temperature, n=n)

    for sampling_params in configs:
        environ["CUDA_VISIBLE_DEVICES"] = "7"
        if sampling_params.get("num_beams", 1) > 4 and "3B" in model:
            environ["CUDA_VISIBLE_DEVICES"] = "7,8"
        print(environ["CUDA_VISIBLE_DEVICES"])
        out_file = run_experiment(sampling_params, llm_params=dict(model_name=model),
                                  force_generation=False, generation_function=run_hf)
        output_files[out_file] = sampling_params
        print()
for k, v in output_files.items():
    print(k, v)
result_files = []
for out_file in output_files:
    # for line in open(Path(output_path, out_file)):
    #     print(json.loads(line)["completion"])
    result_files.append(evaluate_and_save_results(out_file))
sleep(1)
with open(experiments_file, "r") as f:
    experiments = json.load(f)

import matplotlib.pyplot as plt

# Data collection for scatter plot
times = []  # To store time_taken
pass_ks = []  # To store pass@k values
ks = []

times_rep = []
pass_ks_rep = []
ks_rep = []

vllm_pass = []
vllm_times = []
vllm_ks = []

for (out_file, config), result_file in zip(output_files.items(), result_files):
    if "n" in config:
        pass_at_k = calc_pass_at_k_from_results(result_file, [config["n"]])
        time_taken = experiments[str(out_file)]["generation_time"]
        pass_k_value = list(pass_at_k.values())[0]
        vllm_ks.append(config["n"])
        vllm_times.append(time_taken)
        vllm_pass.append(pass_k_value)
    elif "num_beams" in config:
        pass_at_k = calc_pass_at_k_from_results(result_file, [config["num_beams"]])
        time_taken = experiments[str(out_file)]["generation_time"]
        pass_k_value = list(pass_at_k.values())[0]
        ks.append(config["num_beams"])
        times.append(time_taken)
        pass_ks.append(pass_k_value)
        print(f"pass@k {pass_k_value: .2f} ;", f"{round(time_taken)} H100-sec", config)
    else:
        pass_at_k = calc_pass_at_k_from_results(result_file, [config["num_return_sequences"]])
        time_taken = experiments[str(out_file)]["generation_time"]
        pass_k_value = list(pass_at_k.values())[0]
        ks_rep.append(config["num_return_sequences"])
        times_rep.append(time_taken)
        pass_ks_rep.append(pass_k_value)

# Scatter plot
plt.scatter(times, pass_ks, marker="+", label="HF beam search")
plt.xlabel("Time Taken (H100-sec)")
plt.ylabel("pass@k")
plt.title("Scatter Plot of pass@k vs Time Taken")
plt.plot(vllm_times, vllm_pass, label="vLLM repeated sampling")
plt.plot(times_rep, pass_ks_rep, label="HF repeated sampling")
plt.legend()
plt.xscale("log")
plt.savefig("out.png")  # print time and pass at k so we can look at the plot and compare performance

plt.figure()
plt.plot(vllm_ks, vllm_pass, label="vLLM repeated sampling")
plt.plot(ks_rep, pass_ks_rep, label="HF repeated sampling")
plt.scatter(ks, pass_ks, marker="+", label="HF beam search")
plt.legend()
plt.xscale("log")
plt.savefig("out2.png")
