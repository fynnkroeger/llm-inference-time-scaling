from human_eval.evaluation import evaluate_functional_correctness
from inference import out_path


def evaluate_test_cases(file_name: str) -> None:
    eval_result_path = out_path / f"{file_name}_results.jsonl"  # hardcoded in humaneval
    if eval_result_path.exists():
        return
    evaluate_functional_correctness(str(out_path / file_name), n_workers=16)
