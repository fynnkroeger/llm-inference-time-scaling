from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from time import time
from human_eval import evaluation
from os import environ
import uuid
import json
import torch
from vllm_parameter_experiments.inference import run_experiment, plots_path
from vllm_parameter_experiments.run_eval import evaluate_and_save_results, calc_pass_at_k_from_results

experiment_path = Path("/raid/shared/llm-inference-scaling/vllm_parameter_experiments")
output_path = experiment_path / "outputs"
output_path.mkdir(exist_ok=True, parents=True)
experiments_file = experiment_path / "_experiments.json"
plots_path = experiment_path / "plots"
plots_path.mkdir(exist_ok=True, parents=True)


# The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
# The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
def run_hf_beam(out_file, sampling_params, llm_params):
    problems = read_problems()
    prompts = [problem["prompt"] for problem in problems.values()]
    task_ids = list(problems.keys())

    tokenizer = AutoTokenizer.from_pretrained(llm_params["model_name"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(llm_params["model_name"],
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", )
    model.to("cuda")
    samples = []
    t0 = time()

    input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
    # Generate text using beam search
    beam_output = model.generate(
        input_ids.to("cuda"),
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
    environ["CUDA_VISIBLE_DEVICES"] = "3"  # todo do this differently
    environ["TOKENIZERS_PARALLELISM"] = "true"

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
                print("experiment already performed, skipping")
                return file_name

    name = f"{uuid.uuid4()}.jsonl"  # choose out file name randomly
    out_file = output_path / name
    generation_time = run_hf_beam(out_file, sampling_params, llm_params)

    # write only when completed
    experiments[name] = dict(
        sampling_params=sampling_params,
        llm_params=llm_params,
        generation_time=generation_time,
    )
    with open(experiments_file, "w") as f:
        json.dump(experiments, f, indent=4)
    return out_file


if __name__ == "__main__":
    output_files = {}

    models = ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]
    for temperature in [0.6, 1.0]:
        for width in [4]:
            for model in models:
                sampling_params = dict(temperature=temperature, max_new_tokens=10,
                                       num_beams=width, num_return_sequences=width, no_repeat_ngram_size=3,
                                       )
                # num_beam_groups, diversity_penalty
                # repetition_penalty

                out_file = run_experiment(sampling_params, llm_params=dict(model_name=model), force_generation=True)
                output_files[out_file] = dict(temperature=temperature, model=model, beam_width=width, )
                print("done", temperature, width)
    result_files = []
    for out_file in output_files:
        result_files.append(evaluate_and_save_results(out_file))

    for (out_file, config), result_file in zip(output_files.items(), result_files):
        pass_at_k = calc_pass_at_k_from_results(result_file, [4])
        print(pass_at_k)
# print time and pass at k so we can look at the plot and compare performance
