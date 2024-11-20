from rl.llm_generation_cache import HumanEvalLLMGenerationCache

cache = HumanEvalLLMGenerationCache.from_disk()

print(cache.cache.keys())

llm = "meta-llama/Llama-3.2-1B"

llm_cache = cache.cache[llm]

task_ids = llm_cache.keys()

def sampling_to_cache_key(temperature: float):
    params = {
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": 128
    }
    return cache.tranform_sampling_parameters_into_dict_key(params)

temperatures = [i/10 for i in range(0, 25)]

temperature_p_averages = {}
for t in temperatures:
    p_average = 0
    n = 0
    sampling_param_key = sampling_to_cache_key(t)
    for task_id in task_ids:
        result = llm_cache[task_id][sampling_param_key]
        p_average += result.num_passed_samples / (result.num_failed_samples + result.num_passed_samples)
        n += result.num_failed_samples + result.num_passed_samples
    p_average /= len(task_ids)

    print(t, p_average, n / len(task_ids))

