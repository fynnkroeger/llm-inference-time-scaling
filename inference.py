from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from time import time
from human_eval import evaluation
from os import environ
import uuid
import json


def run_generation(out_file, sampling_params, llm_params):
    problems = read_problems()
    prompts = [problem["prompt"] for problem in problems.values()]
    task_ids = list(problems.keys())

    llm = LLM(**llm_params)
    t0 = time()
    outputs = llm.generate(prompts, SamplingParams(**sampling_params))
    print(f"generation time: {time() - t0:.3f}")

    samples = []
    for tid, output in zip(task_ids, outputs):
        for out in output.outputs:
            samples.append(dict(task_id=tid, completion=out.text))
    write_jsonl(out_file, samples)


def run_experiment(sampling_params, llm_params, evaluate=False):
    environ["CUDA_VISIBLE_DEVICES"] = "3"  # todo do this differently
    environ["TOKENIZERS_PARALLELISM"] = "true"

    out_path = Path("/raid/shared/llm-inference-scaling/outputs")
    out_path.mkdir(exist_ok=True, parents=True)

    name = f"{uuid.uuid4()}.jsonl"  # choose out file name randomly
    this_params = dict(sampling_params=sampling_params, llm_params=llm_params)
    experiments_file = out_path / "_experiments.json"
    if experiments_file.exists():
        with open(experiments_file, "r") as f:
            experiments = json.load(f)
    else:
        experiments = {}
    if this_params in experiments.values():
        return
    experiments[name] = this_params
    out_file = out_path / name
    run_generation(out_file, sampling_params, llm_params)
    with open(experiments_file, "w") as f:
        json.dump(experiments, f, indent=4)  # write only when completed
        # todo write time taken
    if evaluate:
        evaluation.evaluate_functional_correctness(
            str(out_file), k=[1, 4, 16, 64, 256]
        )
    return out_file


if __name__ == "__main__":
    sampling_params = dict(temperature=0.8, top_p=0.95, n=4, max_tokens=128)
    llm_params = dict(model="meta-llama/Llama-3.2-1B", gpu_memory_utilization=0.75)
    run_experiment(sampling_params, llm_params, evaluate=True)
