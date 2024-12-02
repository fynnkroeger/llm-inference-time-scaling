from inference import run_experiment
from run_eval import evaluate_and_save_results, calc_pass_at_k_from_results
from inference import experiments_file
import matplotlib.pyplot as plt
import json

optimal = [
    ("meta-llama/Llama-3.1-8B", 0.6),
    ("meta-llama/Llama-3.2-3B", 0.7),
    ("meta-llama/Llama-3.2-1B", 0.7),
]
n_values = [2 ** i for i in range(9)]
# INFO 12-02 23:37:40 config.py:1021] Chunked prefill is enabled with max_num_batched_tokens=512.
output_files = {}
for model, t in optimal:
    for n in n_values:
        print(model, n)
        sampling_params = dict(temperature=t, n=n, max_tokens=128)
        llm_params = dict(model=model, gpu_memory_utilization=0.75)
        out_file = run_experiment(sampling_params, llm_params, evaluate=False)
        output_files[out_file] = dict(temperature=t, model=model, n=n)

result_files = []
for out_file in output_files:
    result_files.append(evaluate_and_save_results(out_file))

print(output_files)

with open(experiments_file, "r") as f:
    experiments = json.load(f)
fig, ax = plt.subplots()
for model, t in optimal:
    times = []
    for (out_file, config), result_file in zip(output_files.items(), result_files):
        if config["model"] == model and config["temperature"] == t:
            print(experiments[out_file])
            time_taken = experiments[out_file]["generation_time"]
            times.append(time_taken)
        # use pass at k from biggest run as better estimation?
        if config["n"] == max(n_values):
            passes = list(calc_pass_at_k_from_results(result_file, n_values).values())
    ax.plot(times, passes, label=model)
    print(model)
ax.set_ylabel("HumanEval pass@k")
ax.set_xlabel("time [s]")
ax.set_xscale("log")
ax.legend()
plt.savefig("time.png", dpi=300)

# another plot with k and time
