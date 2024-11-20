import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rl.llm_generation_cache import HumanEvalLLMGenerationCache
from human_eval.data import read_problems

problems = read_problems()

class AdaptiveTemperatureEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, model: str="meta-llama/Llama-3.2-1B", max_problems: int=164, temperature_start: int=0, temperature_end: int=15, max_iterations: int=1000):
        super().__init__()

        temperature_space = [i/10 for i in range(temperature_start, temperature_end + 1)]
        
        self.action_space = spaces.Discrete(len(temperature_space))
        self._action_to_temperature = {
            i: t for i, t in enumerate(temperature_space)
        }

        self._max_problems = max_problems
        self._solved_problems = {}
        self._time = 0
        self._max_iterations = max_iterations

        self.observation_space = spaces.Dict({
            #"num_solved_problems": spaces.Box(0, max_problems, dtype=int),
            "time": spaces.Box(1, max_iterations, dtype=int)
        })

        self.llm = HumanEvalLLMGenerationCache.from_disk()
        self.llm.use_model(model)


    def _get_obs(self):
        return {"time": np.array(self._time)} #{"num_solved_problems": np.array([len(self._solved_problems)]), "time": np.array(self._time)}
    
    def step(self, action):
        
        temperature = self._action_to_temperature[action]
        non_solved_task_ids, _ = self.get_prompts_for_non_solved_problems(self._solved_problems)
        generation_results = self.llm.generate(non_solved_task_ids, {
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": 128,
            "n": 1,
        })
        newly_solved_problems = {task_id: passed for task_id, passed in generation_results.items() if passed}
        self._solved_problems = self._solved_problems | newly_solved_problems
        self._time += 1

        from math import exp
        num_previously_solved_problems = len(self._solved_problems) - len(newly_solved_problems)
        reward = len(newly_solved_problems)#len(self._solved_problems)#- (self._max_problems - len(self._solved_problems)) # + sum([i + num_previously_solved_problems for i in range(1, len(newly_solved_problems) + 1)])
        observation = self._get_obs()
        terminated = len(self._solved_problems) >= self._max_problems or self._time > self._max_iterations
        truncated = False
        info = {}


        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._solved_problems = {}
        self._time = 0

        observation = self._get_obs()
        return observation, {}

    def get_prompts_for_non_solved_problems(self, solved_problems: dict[str, bool]
):
        prompts = []
        task_ids = []
        for task_id in problems:
            if task_id not in solved_problems:
                prompt = problems[task_id]["prompt"]
                task_ids.append(task_id)
                prompts.append(prompt)
        return task_ids, prompts