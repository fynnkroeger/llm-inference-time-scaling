from inference import run_experiment
from run_eval import evaluate_and_save_results, calc_pass_at_k_from_results
import matplotlib.pyplot as plt

output_files = {}
# run experiments
models = ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]
for temperature in [0.6, 0.8, 1, 1.2]:
    for model in models:
        sampling_params = dict(temperature=temperature, n=256, max_tokens=128)
        llm_params = dict(model=model, gpu_memory_utilization=0.75)
        out_file = run_experiment(sampling_params, llm_params, evaluate=False)
        output_files[out_file] = dict(temperature=temperature, model=model)

# run evals
result_files = []
for out_file in output_files:
    result_files.append(evaluate_and_save_results(out_file))

# generate graphs / tables / ...
k_values = range(1, 256 + 1)

for model in models:
    fig, ax = plt.subplots()
    for (out_file, config), result_file in zip(output_files.items(), result_files):
        if config["model"] != model:
            continue
        pass_at_k = calc_pass_at_k_from_results(result_file, k_values)
        ax.plot(
            k_values, list(pass_at_k.values()), label=f"temp {config['temperature']}"
        )
    model_name_clean = model.split("/")[1]
    ax.set_title(model_name_clean)
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.set_ylabel("HumanEval pass@k")
    ax.set_xlabel("k")
    plt.savefig(f"{model_name_clean}.png")
