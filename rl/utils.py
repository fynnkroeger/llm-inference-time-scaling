import json

def read_samples(file_path: str) -> list:
    data = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def get_task_ids_and_prompts_for_non_solved_problems(
    solved_problems: dict[str, bool], problems: dict
) -> tuple[list[str], list[str]]:
    prompts = []
    task_ids = []
    for task_id in problems:
        if task_id not in solved_problems:
            prompt = problems[task_id]["prompt"]
            task_ids.append(task_id)
            prompts.append(prompt)
    return task_ids, prompts

from rl.runner import evaluate_only_functional_correctness

def judge_problems(outputs: list, task_ids: list[str]) -> tuple[dict[str, bool], list]:
    """
    samples = []
    for task_id, output in zip(task_ids, outputs):
        for i in range(len(output.outputs)):
            generated_text = output.outputs[i].text
            samples.append(dict(task_id=task_id, completion=generated_text))
    """
    results = evaluate_only_functional_correctness(outputs, n_workers=64)

    solved_problems = {}
    for result in results:
        if result["passed"]:
            solved_problems[result["task_id"]] = True
            
    return solved_problems, results