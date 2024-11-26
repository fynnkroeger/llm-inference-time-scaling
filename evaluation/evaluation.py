from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import tqdm

from human_eval.data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from human_eval.execution import check_correctness


def evaluate_only_functional_correctness(
    samples: list[dict],
    out_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
) -> list[dict]:
    """
    Evaluates the functional correctness of generated samples and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)

    # Preload samples to reduce I/O
    n_samples = len(samples)

    # Initialize results storage
    completion_id = Counter()
    results = defaultdict(list)

    # Run test cases in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        # Prepare and submit tasks in batches to reduce I/O overhead
        futures = {
            executor.submit(
                check_correctness,
                problems[sample["task_id"]],
                sample["completion"],
                timeout,
                completion_id[sample["task_id"]]
            ): (sample["task_id"], completion_id[sample["task_id"]])
            for sample in samples
        }

        # Collect results as they complete
        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=n_samples):
            task_id, comp_id = futures[future]
            try:
                result = future.result()
                results[task_id].append((comp_id, result))
            except Exception as e:
                print(f"Error processing task {task_id}: {e}")


    # Combine results and save them in one go
    def combine_results():
        for sample in samples:
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample
    
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))
    