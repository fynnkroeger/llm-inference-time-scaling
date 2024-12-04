from shared_utils.code_evaluation.utils import read_samples
from collections import defaultdict
import seaborn as sns
import math

df_data = []

for temperature in ["0.4-with-function-outputs", "0.8-with-expected-duplicates","0.6-with-expected-duplicates" ,"0.4-with-expected-duplicates"]:
    data = read_samples(f"./outputs/samples-t{str(temperature)}.jsonl")

    problems = defaultdict(lambda: [])
    for x in data:
        problems[x["task_id"]].append(x)

    solved_at_time_t = defaultdict(lambda: 0)
   

    for task_id, solutions in problems.items():
        unique_solutions = set()
        algorithm_functional_outputs = set()
        collisions = 0
        is_task_solved = False
        p_is_collision = 0
        num_of_generations_with_an_error = 0

        for i, solution in enumerate(solutions):
            algorithm_functional_outputs.add(tuple(solution["function_outputs"]))

            if solution["completion"] in unique_solutions:
                collisions += 1
            else:
                unique_solutions.add(solution["completion"])
                p_is_collision += math.exp(solution["cumulative_logprob"])
            if not is_task_solved and solution["passed"]:
                solved_at_time_t[i] += 1
                is_task_solved = True
            
            for f_output in solution["function_outputs"]:
                if isinstance(f_output, str):
                    num_of_generations_with_an_error += 1
                    break
            
            df_data.append({
                "timestep": i,
                "unique_completions": len(unique_solutions),
                "unique_algorithms": len(algorithm_functional_outputs),
                "temperature": temperature,
                "p_algoritm_is_duplicate": 1 - len(algorithm_functional_outputs) / (i + 1),
                "num_of_generations_with_an_error": num_of_generations_with_an_error
            })


import pandas as pd
df = pd.DataFrame.from_records(df_data)

for y in [ "unique_algorithms", "p_algoritm_is_duplicate", "num_of_generations_with_an_error"]:#["unique_completions", "collisions", "num_solved_problems", "p_is_collision"]:
    plot = sns.lineplot(df, x="timestep", y=y, hue="temperature")

    plot.figure.savefig(f"./outputs/temperature_analysis/{y}.png", dpi=300)
    plot.cla()