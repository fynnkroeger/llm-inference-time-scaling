from ctypes import Union
import dask.bag as db
import json
import math
from typing import Union
# cat outputs/results2.jsonl | nl -s "," -w 1 > outputs/results_with_lines.jsonl
from enum import Enum

N_SAMPLES_PER_TASK = 1000

class ResultCatogires(str, Enum):
    passed = "passed"
    expected_colon = "expected colon"
    unterminated_triple_quote = "unterminated triple-quoted string literal"
    other = "other"

def categorize_result(result_message: str) -> ResultCatogires:
    if result_message == "passed":
        return ResultCatogires.passed
    elif "expected ':'" in result_message:
        return ResultCatogires.expected_colon
    elif "failed: unterminated triple-quoted string literal" in result_message:
        return ResultCatogires.unterminated_triple_quote
    else:
        return ResultCatogires.other
    

if __name__ == "__main__":
    b = db.read_text("outputs/results_with_lines.jsonl")
    b = b.map(lambda x: x.split(",", 1))
    b = b.map(lambda x: (int(x[0]), json.loads(x[1])))
    
    def get_task_run(x: tuple[int, dict]) -> int:
        # task_id: HumanEval/0
        task_index = int(x[1]["task_id"].split("/")[1])
        return x[0] - task_index * N_SAMPLES_PER_TASK
    
    def get_pass_k(x: Union[tuple[int, dict], tuple[int, bool]]) -> tuple[int, bool]:
        if isinstance(x, tuple) and isinstance(x[1], bool):
            return x[0], x[1]
        return get_task_run(x), x[1]["passed"]
    
    def passed_at(x: tuple[int, dict], y: tuple[int, dict]) -> tuple[int, bool]:
        x_samples, x_is_passed = get_pass_k(x)
        y_samples, y_is_passed = get_pass_k(y)
        if x_is_passed and y_is_passed:
            return min(x_samples, y_samples), True
        elif x_is_passed:
            return x_samples, True
        elif y_is_passed:
            return y_samples, True
        else:
            return max(x_samples, y_samples), False
    
    g = b.foldby(lambda x: x[1]["task_id"], passed_at, initial=(0, False))
    g = g.map(lambda x: {"task_id": x[0], "num_samples": x[1][0], "was_solved": x[1][1]})
    results = g.compute()
    print(results)
    with open("outputs/analyzed_results.json", "w") as file:
        json.dump(results, file)
