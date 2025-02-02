import json
from pathlib import Path
from pprint import pprint
from collections import Counter

experiment_path = Path("/raid/shared/llm-inference-scaling/vllm_parameter_experiments")

with open(experiment_path / "_experiments.json", "r") as f:
    experiments = json.load(f)

pprint(experiments)
print(len(experiments))

# print("vllm", [experiments])
print(Counter(tuple(e["llm_params"].keys()) for e in experiments.values()))
# input("enter to continue deleting")
filtered_exp = {}
for key, settings in experiments.items():
    if "model_name" in settings["llm_params"] or "tensor_parallel_size" in settings["llm_params"]:
        (experiment_path / "outputs" / key).unlink()
    else:
        filtered_exp[key] = settings
print(len(filtered_exp))
with open(experiment_path / "_experiments.json", "w") as f:
    json.dump(filtered_exp, f)
