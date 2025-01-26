import os

from vllm import LLM
from vllm.sampling_params import BeamSearchParams
from human_eval.data import write_jsonl, read_problems
from time import time

from vllm_parameter_experiments.inference import run_experiment, plots_path
from vllm_parameter_experiments.run_eval import evaluate_and_save_results, calc_pass_at_k_from_results

output_files = {}


def run_generation_vllm_beam(out_file, sampling_params, llm_params):
    problems = read_problems()
    prompts = [problem["prompt"] for problem in problems.values()]
    task_ids = list(problems.keys())

    llm = LLM(**llm_params)
    t0 = time()
    outputs = llm.beam_search(prompts, BeamSearchParams(**sampling_params))
    generation_time = time() - t0
    samples = []
    for tid, output in zip(task_ids, outputs):
        for out in output.sequences:
            samples.append(dict(task_id=tid, completion=out.text))
    write_jsonl(out_file, samples)
    return generation_time


models = ["meta-llama/Llama-3.2-1B"]
for temperature in [0.6]:
    for width in [2, 4]:
        for model in models:
            os.environ["CUDA_VISIBLE_DEVICES"] = "7"
            sampling_params = dict(temperature=temperature, beam_width=width, max_tokens=128)
            llm_params = dict(model=model, gpu_memory_utilization=0.75)
            out_file = run_experiment(sampling_params, llm_params, generation_function=run_generation_vllm_beam,
                                      force_generation=True)
            output_files[out_file] = dict(temperature=temperature, model=model, beam_width=width)
            print("done", temperature, width)

result_files = []
for out_file in output_files:
    result_files.append(evaluate_and_save_results(out_file))

for (out_file, config), result_file in zip(output_files.items(), result_files):
    pass_at_k = calc_pass_at_k_from_results(result_file, [4])
    print(pass_at_k)
