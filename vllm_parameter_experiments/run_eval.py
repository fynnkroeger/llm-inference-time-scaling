from pathlib import Path
from human_eval.evaluation import estimate_pass_at_k
from vllm_parameter_experiments.inference import out_path
import numpy as np
import json
from human_eval.data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from human_eval.execution import check_correctness
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm


def evaluate_functional_correctness(
        sample_file: str,
        n_workers: int = 128,
        timeout: float = 3.0,
        problem_file: str = HUMAN_EVAL,
):
    """
    refactor of the OpenAI implementation that only writes the results
    """

    problems = read_problems(problem_file)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, combine_results())


def evaluate_and_save_results(file_name: str) -> Path:
    eval_result_path = out_path / f"{file_name}_results.jsonl"  # hardcoded in humaneval
    if eval_result_path.exists():
        print("eval already performed, skipping")
        return eval_result_path
    evaluate_functional_correctness(str(out_path / file_name))
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
