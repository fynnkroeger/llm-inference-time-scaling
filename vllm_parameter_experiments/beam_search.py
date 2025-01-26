from vllm_parameter_experiments.inference import run_experiment, plots_path
from vllm_parameter_experiments.run_eval import evaluate_and_save_results, calc_pass_at_k_from_results

output_files = {}

models = ["meta-llama/Llama-3.2-1B"]
for temperature in [0.6]:
    for width in [4]:
        for model in models:
            sampling_params = dict(temperature=temperature, beam_width=width, max_tokens=128)
            llm_params = dict(model=model, gpu_memory_utilization=0.75)
            out_file = run_experiment(sampling_params, llm_params, beam_search=True)
            output_files[out_file] = dict(temperature=temperature, model=model, beam_width=width)
            print("done", temperature, width)

result_files = []
for out_file in output_files:
    result_files.append(evaluate_and_save_results(out_file))

for (out_file, config), result_file in zip(output_files.items(), result_files):
    pass_at_k = calc_pass_at_k_from_results(result_file, [4])
    print(pass_at_k)
