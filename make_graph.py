import json
from pathlib import Path
from human_eval.evaluation import estimate_pass_at_k
import numpy as np
import matplotlib.pyplot as plt

ks = range(1, 256 + 1)

out_path = Path("/raid/shared/llm-inference-scaling/outputs")
experiments_file = out_path / "_experiments.json"
with open(experiments_file, "r") as f:
    experiments = json.load(f)


def calc_pass_at_k_from_results(result_file, ks):
    total = np.zeros((164,))
    correct = np.zeros((164,))
    for line in result_file.read_text().splitlines():
        loaded = json.loads(line)
        index = int(loaded["task_id"].split("/")[1])
        total[index] += 1
        correct[index] += loaded["passed"]
    pass_at_k = {
        k: estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }
    return pass_at_k


fig, ax = plt.subplots()

for name, experiment_settings in experiments.items():
    match experiment_settings:
        case {
            "sampling_params": {"temperature": t, "n": 256, "max_tokens": 128},
            "llm_params": {"model": "meta-llama/Llama-3.2-1B"},
        }:
            print(t)
            print(name)
            result_file = out_path / f"{name}_results.jsonl"
            if not result_file.exists():
                continue

            res = calc_pass_at_k_from_results(result_file, ks)
            ax.plot(ks, list(res.values()), label=f"temperature {t}")
            print(res)
ax.set_title("Llama-3.2-1B")

ax.set_xscale("log", base=2)
ax.legend()
ax.set_ylabel("HumanEval pass@k")
ax.set_xlabel("k")
plt.savefig("out.png")
