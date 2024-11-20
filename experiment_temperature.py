from inference import run_experiment

for temperature in [0.6, 0.8, 1, 1.2]:
    sampling_params = dict(temperature=temperature, n=256, max_tokens=128)
    llm_params = dict(model="meta-llama/Llama-3.1-8B", gpu_memory_utilization=0.75)
    run_experiment(sampling_params, llm_params, evaluate=False)
