from shared_utils.code_evaluation.utils import read_samples
from collections import defaultdict
import seaborn as sns
import math

df_data = []
max_t = 256


def extract_num_solved_problems_from_logs(log_file: str) -> dict[int, int]:
    with open(log_file, "r") as file:
        lines = file.readlines()
    num_solved_problems = 0

    solved_at_time_t = {}
    for line in lines:
        if "Newly solved problems: " in line:
            iteration = int(line.removeprefix("K: ").split(" ")[0])
            newly_solved_problems = int(line.split(" ")[7][:-1])
            num_solved_problems += newly_solved_problems
            solved_at_time_t[iteration] = num_solved_problems
    return solved_at_time_t


#w = model.embed_tokens.weight
# torch.matmul(torch.rand((2048,)), w.T).softmax(dim=-1)
for temperature in ["samples-1B-t0.8-with-hidden-states-value-estimate","samples-1B-t0.8-with-hidden-states-value-estimate-2","samples-1B-t0.8-with-hidden-states-value-estimate-advantage", "samples-t0.8-with-hidden-states"]:
    data = read_samples(f"./outputs/{temperature}.jsonl")

    problems = defaultdict(lambda: [])
    for x in data:
        problems[x["task_id"]].append(x)

    solved_at_time_t = defaultdict(lambda: 0)

    for task_id, solutions in problems.items():
        unique_solutions = set()
        collisions = 0
        is_task_solved = False
        p_is_collision = 0

        for i, solution in enumerate(solutions):
            
            if solution["completion"] in unique_solutions:
                collisions += 1
            else:
                unique_solutions.add(solution["completion"])
                p_is_collision += math.exp(solution["cumulative_logprob"])
            if not is_task_solved and solution["passed"]:
                solved_at_time_t[i] += 1
                is_task_solved = True
                #print(temperature,solved_at_time_t[i], i)
            df_data.append({
                "timestep": i,
                "unique_completions": len(unique_solutions),
                "collisions": collisions,
                "temperature": temperature,
                "p_is_collision": p_is_collision
            })




    num_solved_tasks = 0
    for i in range(max_t):
        num_solved_tasks += solved_at_time_t[i]
        df_data.append({
            "timestep": i,
            "num_solved_problems": num_solved_tasks,
            "temperature": temperature
        })
    print(temperature, max([len(x) for x in problems.values()]))
import pandas as pd

solved_problems = extract_num_solved_problems_from_logs("outputs/value_estimator.log")
for i, n in solved_problems.items():
    if i > max_t:
        break
    df_data.append({
        "timestep": i,
        "num_solved_problems": n,
        "temperature": "with-advantage"
    })

df = pd.DataFrame.from_records(df_data)

for y in [ "num_solved_problems", "unique_completions", "collisions"]:#["unique_completions", "collisions", "num_solved_problems", "p_is_collision"]:
    plot = sns.lineplot(df, x="timestep", y=y, hue="temperature")

    plot.figure.savefig(f"./outputs/value_estimator/{y}.png", dpi=300)
    plot.cla()