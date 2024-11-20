from pathlib import Path
import json
from dataclasses import dataclass
import random
from rl.llm_generation_cache import HumanEvalLLMGenerationCache
from human_eval.data import read_problems
from statistics import mean

problems = read_problems()

llm_cache = HumanEvalLLMGenerationCache.from_disk()
llm_cache.use_model("meta-llama/Llama-3.2-1B")

@dataclass
class TaskCompletion:
    task_id: str
    completion: str
    result: str
    passed: bool


discount_factor = 1


def get_prompts_for_non_solved_problems(
    solved_problems: dict[str, bool]
) -> tuple[list[str], list[str]]:
    prompts = []
    task_ids = []
    for task_id in problems:
        if task_id not in solved_problems:
            prompt = problems[task_id]["prompt"]
            task_ids.append(task_id)
            prompts.append(prompt)
    return task_ids, prompts

import math
def reward(s: tuple[int, int], solved_problems: dict, t:int=0) -> float: 
    return len(solved_problems)

def reward_with_probabilities(s: int, s_i: int, solved_problems: dict, p_already_solved: dict) -> float:
    reward = 0
    for task_id in solved_problems.keys():
        reward += solved_problems[task_id] * (1 - p_already_solved[task_id])
    return reward

MAX_STEPS = 100
def simulate(s_0: tuple[int, int], all_states: list[tuple[int, int]], policy, parametric_q, weights):
    """
    s: int = random.randint(0, 30)
    solved_problems = {}


    all_task_ids, _ = get_prompts_for_non_solved_problems({})
    for _ in range(s):
        def choose_random_task_id():
            random_solved_problem_index = random.randint(0, len(all_task_ids) - 1)
            return all_task_ids[random_solved_problem_index]
        
        random_task_id = choose_random_task_id()
        while random_task_id in solved_problems:
            random_task_id = choose_random_task_id()
        solved_problems[random_task_id] = True

    """
    s = s_0
    solved_problems = {}
    q_target_values = {}

    q_inputs = []
    rewards = []

    n_samples = 10
    for t in range(int(MAX_STEPS / n_samples)):
        if s[0] >= all_states[-1][0]:
            break

        temperature = policy(s)
        print(f"State: {s} Temperature: {temperature}")
        non_solved_task_ids, _ = get_prompts_for_non_solved_problems(solved_problems)
        generation_results = llm_cache.generate(
            non_solved_task_ids,
            {
                "temperature": temperature,
                "top_p": 0.95,
                "max_tokens": 128,
                "n": n_samples,
            },
        )
        newly_solved_problems = {task_id: passed for task_id, passed in generation_results.items() if passed}

        print("Solved the following problems:", newly_solved_problems)
        solved_problems = solved_problems | newly_solved_problems
        r = reward(s,  solved_problems, t)

        s_next =  (s[0] + len(newly_solved_problems), s[1] + 1)
        #r_total = r + discount_factor * parametric_q(s_next, policy(s_next), weights)
        
        rewards.append(r)
        q_inputs.append((s, temperature))

       
        s = s_next

    for i in range(len(q_inputs)):
        new_q_value = sum([(discount_factor ** (j-i)) * rewards[j] for j in range(i,len(q_inputs))])
        if q_inputs[i] in q_target_values:
            q_target_values[q_inputs[i]].append(new_q_value)
        else:
            q_target_values[q_inputs[i]] = [new_q_value]

    for key, q_values in q_target_values.items():
        q_target_values[key] = mean(q_values)


    return q_target_values


def simulate_with_probabilities(s_0: int, all_states: list[int], policy):
    s = s_0
    q_target_values = {}

    all_task_ids, _ = get_prompts_for_non_solved_problems({})
    p_problem_already_solved = {task_id: 0.0 for task_id in all_task_ids}

    q_inputs = []
    rewards = []
    while True:
        s_i = all_states.index(s)
        if s_i == len(all_states) - 1:
            break

        next_s = all_states[s_i + 1]
        n_samples = next_s - s

        temperature = policy(s)
        print(f"State: {s} Temperature: {temperature}")
        generation_results = llm_cache.generate(
            all_task_ids,
            {
                "temperature": temperature,
                "top_p": 0.95,
                "max_tokens": 128,
                "n": n_samples,
            },
            use_expected_value=True
        )

        r = reward_with_probabilities(s, s_i, generation_results, p_problem_already_solved)
        rewards.append(r)

        q_inputs.append((s, temperature))

        for task_id in all_task_ids:
            p_problem_already_solved[task_id] += (1 - p_problem_already_solved[task_id]) * generation_results[task_id]
        s = next_s

    for i in range(len(q_inputs)):
        q_target_values[q_inputs[i]] = rewards[i]

    return q_target_values

"""

Q(s, a) -> 

[]
"""
