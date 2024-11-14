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
    prompts = []
    task_ids = []
    for task_id, problem in problems.items():
        prompts.append(problem["prompt"])
        task_ids.append(task_id)

    llm = LLM(**llm_params)
    t0 = time()
    outputs = llm.generate(prompts, SamplingParams(**sampling_params))
    print(f"generation time: {time() - t0:.3f}")

    samples = []
    for tid, output in zip(task_ids, outputs):
        for out in output.outputs:
            samples.append(dict(task_id=tid, completion=out.text))
    write_jsonl(out_file, samples)


if __name__ == "__main__":
    environ["CUDA_VISIBLE_DEVICES"] = "3"
    environ["TOKENIZERS_PARALLELISM"] = "true"

    out_path = Path("/raid/shared/llm-inference-scaling/outputs")
    out_path.mkdir(exist_ok=True, parents=True)

    # todo dont generate again with same settings

    sampling_params = dict(temperature=0.8, top_p=0.95, n=1, max_tokens=128)
    llm_params = dict(model="meta-llama/Llama-3.2-1B", gpu_memory_utilization=0.75)

    name = f"{uuid.uuid4()}.jsonl"  # choose out file name randomly
    with open(out_path / "_experiments.jsonl", "a") as f:
        info = dict(name=name, sampling_params=sampling_params, llm_params=llm_params)
        f.write(json.dumps(info)+"\n")

    out_file = out_path / name
    run_generation(
        out_file, sampling_params, llm_params
    )

    result = evaluation.evaluate_functional_correctness(
        str(out_file), k=[1, 4, 16, 64, 256]
    )
    print(result)
