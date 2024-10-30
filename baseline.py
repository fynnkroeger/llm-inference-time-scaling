from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from pathlib import Path

problems = read_problems()
samples = []
prompts = []
for task_id in problems:
    prompt = problems[task_id]["prompt"]
    prompts.append(prompt)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="meta-llama/Llama-3.2-1B")
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    samples.append(dict(task_id=task_id, completion=generated_text))
out_path = Path("outputs")
out_path.mkdir(exist_ok=True)
write_jsonl(out_path / "samples.jsonl", samples)