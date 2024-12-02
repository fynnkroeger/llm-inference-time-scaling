from inference import run_experiment
from run_eval import evaluate_test_cases


output_files = []
# run experiments
for temperature in [0.6, 0.8, 1, 1.2]:
    sampling_params = dict(temperature=temperature, n=256, max_tokens=128)
    llm_params = dict(model="meta-llama/Llama-3.1-1B", gpu_memory_utilization=0.75)
    out_file = run_experiment(sampling_params, llm_params, evaluate=False)
    output_files.append(out_file)

# run evals
for out_file in output_files:
    evaluate_test_cases(out_file)
