import seaborn as sns
from gymnasium import Env
from stable_baselines3.common.policies import BasePolicy
import pandas as pd

def plot_policy(env: Env, *policies: BasePolicy, plot_file_name: str="policy_comparisons.png", n_eval_episodes: int=20):
    truncated = False
    terminated = False
    
    print(policies)
    data = []
    for policy in policies:
        
        print(policy._get_name())
        for _ in range(n_eval_episodes):
            obs = env.reset()
            t = 0
            total_reward = 0.0
            finished = False
            while not finished:
                action, _ = policy.predict(obs) # type: ignore
                obs, reward, finished, _ = env.step(action)
                total_reward += reward[0] # type: ignore
                data.append({
                    "time": t,
                    "reward": total_reward,
                    "policy": policy._get_name()
                })
                t += 1

    df = pd.DataFrame.from_records(data)

    print(df["policy"].unique())
    print(df["time"].max())
    plot = sns.lineplot(df, x="time", y="reward", hue="policy")
    plot.set_xscale("log")
    pos = plot.get_position()
    plot.set_position([pos.x0, pos.y0, pos.width * 0.7, pos.height])
    
    plot.legend(loc='center right', bbox_to_anchor=(1.65, 0.5))
    plot.get_figure().savefig(plot_file_name) # type: ignore


    
        
