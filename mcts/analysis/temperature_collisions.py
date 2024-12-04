from shared_utils.code_evaluation.utils import read_samples
from collections import defaultdict
import seaborn as sns
import math

df_data = []

for temperature in [0.4, "0.8-with-expected-duplicates-long",0.8, 1.2]:
    data = read_samples(f"./outputs/samples-t{str(temperature)}.jsonl")

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




    max_t = 1024
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
df = pd.DataFrame.from_records(df_data)

for y in [ "num_solved_problems"]:#["unique_completions", "collisions", "num_solved_problems", "p_is_collision"]:
    plot = sns.lineplot(df, x="timestep", y=y, hue="temperature")

    plot.figure.savefig(f"./outputs/temperature_analysis/{y}.png", dpi=300)
    plot.cla()