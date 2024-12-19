from vllm_parameter_experiments.inference import run_experiment, plots_path
from vllm_parameter_experiments.run_eval import evaluate_and_save_results, calc_pass_at_k_from_results

output_files = {}

models = ["meta-llama/Llama-3.2-1B"]
for temperature in [0.6, 0.8, 1]:
    for width in [2, 8, 32]:
        for model in models:
            sampling_params = dict(temperature=temperature, beam_width=width, max_tokens=128)
            llm_params = dict(model=model, gpu_memory_utilization=0.75)
            out_file = run_experiment(sampling_params, llm_params, evaluate=False, beam_search=True)
            output_files[out_file] = dict(temperature=temperature, model=model, beam_width=width)
            print("done", temperature, width)

result_files = []
for out_file in output_files:
    result_files.append(evaluate_and_save_results(out_file))

# todo compare with other
