from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from time import time
from human_eval import evaluation

def run_generation():
    problems = read_problems()
    samples = []
    prompts = []
    task_ids = []
    for task_id in problems:
        prompt = problems[task_id]["prompt"]
        prompts.append(prompt)
        task_ids.append(task_id)


    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, n=256, max_tokens=128)
    llm = LLM(model="meta-llama/Llama-3.2-1B", gpu_memory_utilization=0.75)
    t0 = time()
    outputs = llm.generate(prompts, sampling_params)
    print(f"generation time: {time() - t0:.3f}")

    for tid, output in zip(task_ids, outputs):
        # prompt = output.prompt
        for out in output.outputs:
            samples.append(dict(task_id=tid, completion=out.text))
    write_jsonl(out_file, samples)

if __name__ == "__main__":
    out_path = Path("/raid/shared/llm-inference-scaling/outputs")
    out_path.mkdir(exist_ok=True, parents=True)
    out_file = out_path / "samples.jsonl"
    # todo dont generate again with same settings
    run_generation()

    result = evaluation.evaluate_functional_correctness(str(out_file), k=[1, 4, 16, 64, 256], n_workers=32)
    print(result)