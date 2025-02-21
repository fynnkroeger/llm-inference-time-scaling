from pathlib import Path
import matplotlib.pyplot as plt

experiment_path = Path("/raid/shared/llm-inference-scaling/prefix_sampling_experiments")

prefix_internal_times_70 = [5274.3930876255035,  4582.992676496506, 3807.308326482773, 3239.214138031006, 3368.595718383789]
base_internal_times_70 = [6449.06245470047,  5581.008045911789, 3879.9213721752167, 3259.7462706565857, 3310.6455211639404]

prefix_internal_times_8 = [2833.4032640457153, 3564.3958008289337, 3288.6198654174805, 3095.996219396591, 3329.683702468872]
base_internal_times_8 = [3455.1942121982574, 3874.902323961258, 3367.249425172806, 3148.980687856674, 3364.6879568099976]

temps = [0.2, 0.4, 0.6, 0.8, 1.0]

percentages_70 = [b / a for a, b in zip(prefix_internal_times_70, base_internal_times_70)]
percentages_8 = [b / a for a, b in zip(prefix_internal_times_8, base_internal_times_8)]



plt.figure(figsize=(10, 6))
plt.plot(temps, percentages_70, marker='o', linestyle='-', color='b', label='70B Model')
plt.plot(temps, percentages_8, marker='x', linestyle='--', color='r', label='8B Model')
plt.xlabel('Temperature')
plt.ylabel('Prefix Caching Time / Baseline Time')
plt.title('Time Gain over Temperature')
plt.legend()
plt.grid(True)

plt.savefig(experiment_path / f"time_gain_over_temp.png")