import seaborn as sns
import pandas as pd

errors = [{
    "training_dataset_size": 20_000,
    "mean_error": 0.4572383099471834,
    "mean_random_baseline_error": 0.4903127363726173,
    "model": "2 layer MLP",
},
{
    "training_dataset_size": 40_000,
    "mean_error": 0.45257314665578185,
    "mean_random_baseline_error": 0.4901630799217903,
    "model": "2 layer MLP",
},
{
    "training_dataset_size": 80_000,
    "mean_error": 0.44709959206875555,
    "mean_random_baseline_error": 0.4899148274188829,
    "model": "2 layer MLP",
},
{
    "training_dataset_size": 160_000,
    "mean_error": 0.4463769494091731,
    "mean_random_baseline_error": 0.489818139366639,
    "model": "2 layer MLP",
},
{
    "training_dataset_size": 320_000,
    "mean_error": 0.4429596602226437,
    "mean_random_baseline_error": 0.4896337514318198,
    "model": "2 layer MLP",
},
{
    "training_dataset_size": 550_284,
    "mean_error": 0.4508082362878476,
    "mean_random_baseline_error": 0.4904330820727314,
    "model": "2 layer MLP",
},
{
    "training_dataset_size": 20_000,
    "mean_error": 0.4903127363726173,
    "model": "random baseline",
},
{
    "training_dataset_size": 40_000,
    "mean_error": 0.4901630799217903,
    "model": "random baseline",
},
{
    "training_dataset_size": 80_000,
    "mean_error": 0.4899148274188829,
    "model": "random baseline",
},
{
    "training_dataset_size": 160_000,
    "mean_error": 0.489818139366639,
    "model": "random baseline",
},
{
    "training_dataset_size": 320_000,
    "mean_error": 0.4896337514318198,
    "model": "random baseline",
},
{
    "training_dataset_size": 550_284,
    "mean_error": 0.4904330820727314,
    "model": "random baseline",
}
]

df = pd.DataFrame.from_records(errors)

plt = sns.lineplot(df, x="training_dataset_size", y="mean_error", hue="model", style="model",
    markers=True, dashes=False)
plt.set_xscale("linear")
plt.figure.savefig("mlp_value_estimator.png")