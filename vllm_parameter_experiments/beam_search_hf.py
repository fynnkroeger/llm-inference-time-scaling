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

experiment_path = Path("/raid/shared/llm-inference-scaling/vllm_parameter_experiments")
output_path = experiment_path / "outputs"
output_path.mkdir(exist_ok=True, parents=True)
experiments_file = experiment_path / "_experiments.json"
plots_path = experiment_path / "plots"
plots_path.mkdir(exist_ok=True, parents=True)


def run_hf_beam(out_file, sampling_params, llm_params):
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
            early_stopping=True,
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


def run_experiment(sampling_params, llm_params, force_generation=False):
    environ["TOKENIZERS_PARALLELISM"] = "true"
    environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if experiments_file.exists():
        with open(experiments_file, "r") as f:
            experiments = json.load(f)
        to_delete = []
        for name in experiments:
            if not (output_path / name).exists():
                to_delete.append(name)
        if to_delete:
            for name in to_delete:
                print(f"deleting {name} as file not found")
                del experiments[name]
            with open(experiments_file, "w") as f:
                json.dump(experiments, f, indent=4)
    else:
        experiments = {}
    if not force_generation:  # delete the file?
        for file_name, config in experiments.items():
            if (
                    config["sampling_params"] == sampling_params
                    and config["llm_params"] == llm_params
            ):
                print("experiment already performed, skipping", config)
                return file_name
    print("running experiment", config)
    name = f"{uuid.uuid4()}.jsonl"  # choose out file name randomly
    out_file = output_path / name
    num_gpus_used = len(environ["CUDA_VISIBLE_DEVICES"].split(","))
    generation_time = run_hf_beam(out_file, sampling_params, llm_params) * num_gpus_used
    # write only when completed
    experiments[name] = dict(
        sampling_params=sampling_params,
        llm_params=llm_params,
        generation_time=generation_time,
    )
    with open(experiments_file, "w") as f:
        json.dump(experiments, f, indent=4)
    return out_file


def experiment(sampling_params):
    return run_experiment(sampling_params, llm_params=dict(model_name=model),
                          force_generation=False)


if __name__ == "__main__":
    output_files = {}

    models = ["meta-llama/Llama-3.2-1B"]
    for model in models:
        for width in [2, 4, 6]:  # 16 does not work
            environ["CUDA_VISIBLE_DEVICES"] = "7,8" if ((width > 4) and ("3B" in model)) else "7"
            print(environ["CUDA_VISIBLE_DEVICES"])
            for repetition_penalty in [1.0, 1.1, 1.2]:
                sampling_params = dict(max_new_tokens=128,
                                       num_beams=width, num_return_sequences=width,
                                       repetition_penalty=repetition_penalty,
                                       do_sample=False)
                # normal beam search
                out_file = experiment(sampling_params)
                output_files[out_file] = sampling_params

                # sampling beam search
                for temperature in [0.6, 1.0]:
                    sampling_params = dict(max_new_tokens=128,
                                           num_beams=width, num_return_sequences=width,
                                           repetition_penalty=repetition_penalty, temperature=temperature,
                                           do_sample=True)
                    out_file = experiment(sampling_params)
                    output_files[out_file] = sampling_params
                # running experiment {'sampling_params': {'max_new_tokens': 128, 'num_beams': 4, 'num_return_sequences': 4, 'repetition_penalty': 1.2, 'temperature': 1.0, 'do_sample': False, 'num_beam_groups': 2, 'diversity_penalty': 1.0}, 'llm_params': {'model_name': 'meta-llama/Llama-3.2-1B'}, 'generation_time': 23.160406351089478}
                # /home/fynn.kroeger/miniconda3/envs/inference-time-scaling/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:590: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
                #   warnings.warn(
                # /home/fynn.kroeger/miniconda3/envs/inference-time-scaling/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:595: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.

                # diverse beam search
                for num_beam_groups in [2]:
                    for diversity_penalty in [1.0]:
                        sampling_params = dict(max_new_tokens=128,
                                               num_beams=width, num_return_sequences=width,
                                               repetition_penalty=repetition_penalty, num_beam_groups=num_beam_groups,
                                               diversity_penalty=diversity_penalty, do_sample=False)
                        out_file = experiment(sampling_params)
                        output_files[out_file] = sampling_params
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

for (out_file, config), result_file in zip(output_files.items(), result_files):
    pass_at_k = calc_pass_at_k_from_results(result_file, [config["num_beams"]])
    time_taken = experiments[str(out_file)]["generation_time"]
    pass_k_value = list(pass_at_k.values())[0]
    times.append(time_taken)
    pass_ks.append(pass_k_value)
    print(f"pass@k {pass_k_value: .2f} ;", f"{round(time_taken)} H100-sec", config)

# Scatter plot
plt.scatter(times, pass_ks, label="beam search")
plt.xlabel("Time Taken (H100-sec)")
plt.ylabel("pass@k")
plt.title("Scatter Plot of pass@k vs Time Taken")
plt.ylim(0, 0.9)
plt.xlim(1, 100)
plt.plot([1.19, 8.30, 60.41], [0.118, 0.343, 0.547], "r+", label="repeated sampling")
plt.legend()
plt.xscale("log")
plt.savefig("out.png")  # print time and pass at k so we can look at the plot and compare performance
