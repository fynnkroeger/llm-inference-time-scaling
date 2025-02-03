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

data = []
score_data = []
training_dataset_sizes = [1000, 2000, 4000, 10000, 20_000, 40_000, 80_000, 160_000]

for val_type in ["id", "ood"]:
    for n in training_dataset_sizes:
        df = pd.read_csv(f"outputs/single_value_estimator/error-mlp-{val_type}-1B-{n}.csv")
        df_scores = pd.read_csv(f"outputs/single_value_estimator/scores_mlp-{val_type}-1B-{n}.csv")
        data.append({
            "mean_error": df["error"].mean(),
            "training_dataset_size": n,
            "model": f"2 layer MLP ({val_type})"
        })
        data.append({
            "mean_error": df["random_baselin_error"].mean(),
            "training_dataset_size": n,
            "model": f"random baseline"
        })
        data.append({
            "mean_error": df["value"].sub(df["value"].mean()).abs().mean(),
            "training_dataset_size": n,
            "model": f"static mean value predictor baseline"
        })

        score_data.append({
            "accuracy": df_scores["score"].mean(),
            "training_dataset_size": n,
            "model": f"2 layer MLP ({val_type})"
        })
        score_data.append({
            "accuracy": df_scores["random_baseline_score"].mean(),
            "training_dataset_size": n,
            "model": f"random baseline"
        })

        # df["value"].sub(df["value"].mean()).abs().mean()
b1_errors = [
    {
        "training_dataset_size": 1000,
        "mean_error": 0.41865838855346355,
        "model": "2 layer MLP (id)",
    },
    {
        "training_dataset_size": 1000,
        "mean_error": 0.49226594259710044,
        "model": "random_baseline"
    },
    {
        "training_dataset_size": 2000,
        "mean_error": 0.4040902003109872,
        "model": "2 layer MLP (id)",
    },
    {
        "training_dataset_size": 2000,
        "mean_error": 0.4905145767733132,
        "model": "random_baseline"
    },
    {
        "training_dataset_size": 4000,
        "mean_error": 0.3910637510962357,
        "model": "2 layer MLP (id)",
    },
    {
        "training_dataset_size": 4000,
        "mean_error": 0.48577346931010823,
        "model": "random_baseline"
    }
    ,
    {
        "training_dataset_size": 10000,
        "mean_error": 0.38308503254672055,
        "model": "2 layer MLP (id)",
    },
    {
        "training_dataset_size": 4000,
        "mean_error": 0.49368097074367656,
        "model": "random_baseline"
    }
]
df = pd.DataFrame.from_records(data)
df_scores = pd.DataFrame.from_records(score_data)
plt = sns.lineplot(df, x="training_dataset_size", y="mean_error", hue="model", style="model",
    markers=True, dashes=False)



plt.figure.savefig("mlp_value_estimator.png")
plt.clear()

plt2 = sns.lineplot(df_scores, x="training_dataset_size", y="accuracy", hue="model", style="model",
    markers=True, dashes=False)
plt2.figure.savefig("mlp_accuarcy.png")