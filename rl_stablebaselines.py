from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
from stable_baselines3.common.evaluation import evaluate_policy
from analysis.analyse_rl_policies import plot_policy

# Example for the CartPole environment
register(
    # unique identifier for the env `name-version`
    id="Adaptive-Temperature-LLM-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="rl.adaptive_temperature.gymnasium_env:AdaptiveTemperatureEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=500,
)
from rl.adaptive_temperature.gymnasium_env import AdaptiveTemperatureEnv
 
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy

import numpy as np
import torch
from typing import Optional

class BestGuessPolicy(BasePolicy):
    def __init__(self, vec_env, static_action:Optional[int]=None):
        # Access action space directly from the vec_env
        super(BestGuessPolicy, self).__init__(vec_env.observation_space, vec_env.action_space)
        self.vec_env = vec_env
        self.static_action = static_action


    def _predict(self, observation, deterministic: bool = False):
        t = observation["time"].cpu()[0][0].item()
        
        return torch.tensor(min(3+t, 7))
        
        
class BaselinePolicy(BasePolicy):
    def __init__(self, vec_env, static_action:Optional[int]=None):
        # Access action space directly from the vec_env
        super(BaselinePolicy, self).__init__(vec_env.observation_space, vec_env.action_space)
        self.vec_env = vec_env
        self.static_action = static_action

    def _get_name(self):
        if self.static_action is not None:
            return f"Static temperature: {self.static_action/10}"
        else:
            return "Random action policy"
        
    def _predict(self, observation, deterministic: bool = False):
        if self.static_action is None:
            return torch.tensor(vec_env.action_space.sample())
        else:
            return torch.tensor(self.static_action)
        
from rl.adaptive_temperature.gymnasium_env import AdaptiveTemperatureEnv
vec_env = make_vec_env("Adaptive-Temperature-LLM-v1")
print(vec_env.action_space)

policies = []

guess = BestGuessPolicy(vec_env=vec_env)
policies.append(guess)
print(evaluate_policy(guess, vec_env))
print(evaluate_policy(BaselinePolicy(vec_env=vec_env), vec_env))
for i in range(0, 11, 1):
    static_baseline_policy = BaselinePolicy(vec_env, i)
    print(i, evaluate_policy(static_baseline_policy, vec_env))
    policies.append(static_baseline_policy)

plot_policy(vec_env, *policies)
exit()
model = PPO("MultiInputPolicy", vec_env, verbose=1).learn(100000, log_interval=50)
print(model)
print(evaluate_policy(model, vec_env))

obs = vec_env.reset()
for _ in range(32):
    action, _states = model.predict(obs)
    print(obs, action)
    obs, rewards, dones, info = vec_env.step(action)
