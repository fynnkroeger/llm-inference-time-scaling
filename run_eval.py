from pathlib import Path
from human_eval.evaluation import evaluate_functional_correctness
from time import sleep


def find_and_eval(out_path):
    files = [p for p in out_path.iterdir() if p.is_file()]
    output_files = [p for p in files if len(p.name) == 42 and p.suffix == ".jsonl"]
    print(len(output_files))
    not_evaluated = [p for p in output_files if not Path(f"{p}_results.jsonl").exists()]
    for output_file in not_evaluated:
        print(evaluate_functional_correctness(str(output_file)))


if __name__ == "__main__":
    while True:
        find_and_eval(Path("/raid/shared/llm-inference-scaling/outputs"))
        sleep(5)
