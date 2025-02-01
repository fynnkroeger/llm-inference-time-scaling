import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("scores_df-8396.csv")

df2 = df.groupby("mean_path_depth")["score"].aggregate(["mean", "std" ,"count"])
df2 = df2[(df2["std"] > 0.0) & (df2["count"] >= 5)]
plt.clf()
sns.relplot(df2, x="mean_path_depth", y="mean", size="count").figure.savefig("scores_per_depth.png")
