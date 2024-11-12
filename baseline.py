from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from time import time

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
    prompt = output.prompt
    generated_text = output.outputs[0].text
    samples.append(dict(task_id=tid, completion=generated_text))
out_path = Path("/raid/shared/llm-inference-scaling/outputs")
out_path.mkdir(exist_ok=True, parents=True)
write_jsonl(out_path / "samples.jsonl", samples)
