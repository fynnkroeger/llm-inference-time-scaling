from pathlib import Path
from human_eval.evaluation import evaluate_functional_correctness, estimate_pass_at_k
from inference import out_path
import numpy as np
import json


def evaluate_and_save_results(file_name: str) -> Path:
    eval_result_path = out_path / f"{file_name}_results.jsonl"  # hardcoded in humaneval
    if eval_result_path.exists():
        return eval_result_path
    evaluate_functional_correctness(str(out_path / file_name), n_workers=16)
    assert eval_result_path.exists()
    return eval_result_path


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
