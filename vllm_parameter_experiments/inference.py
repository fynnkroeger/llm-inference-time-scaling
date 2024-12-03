from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from time import time
from human_eval import evaluation
from os import environ
import uuid
import json

experiment_path = Path("/raid/shared/llm-inference-scaling/vllm_parameter_experiments")
output_path = experiment_path / "outputs"
output_path.mkdir(exist_ok=True, parents=True)
experiments_file = experiment_path / "_experiments.json"
plots_path = experiment_path / "plots"
plots_path.mkdir(exist_ok=True, parents=True)


def run_generation(out_file, sampling_params, llm_params):
    problems = read_problems()
    prompts = [problem["prompt"] for problem in problems.values()]
    task_ids = list(problems.keys())

    llm = LLM(**llm_params)
    t0 = time()
    outputs = llm.generate(prompts, SamplingParams(**sampling_params))
    generation_time = time() - t0
    samples = []
    for tid, output in zip(task_ids, outputs):
        for out in output.outputs:
            samples.append(dict(task_id=tid, completion=out.text))
    write_jsonl(out_file, samples)
    return generation_time


def run_experiment(sampling_params, llm_params, evaluate=False):
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
    for file_name, config in experiments.items():
        if (
                config["sampling_params"] == sampling_params
                and config["llm_params"] == llm_params
        ):
            print("experiment already performed, skipping")
            return file_name

    name = f"{uuid.uuid4()}.jsonl"  # choose out file name randomly
    out_file = output_path / name
    generation_time = run_generation(out_file, sampling_params, llm_params)

    # write only when completed
    experiments[name] = dict(
        sampling_params=sampling_params,
        llm_params=llm_params,
        generation_time=generation_time,
    )
    with open(experiments_file, "w") as f:
        json.dump(experiments, f, indent=4)
    if evaluate:
        evaluation.evaluate_functional_correctness(str(out_file), k=[1, 4, 16, 64, 256])
    return out_file


if __name__ == "__main__":
    sampling_params = dict(temperature=0.6, n=256, max_tokens=128)
    llm_params = dict(model="meta-llama/Llama-3.1-8B", gpu_memory_utilization=0.75)
    run_experiment(sampling_params, llm_params, evaluate=False)
