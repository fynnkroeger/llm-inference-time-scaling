from shared_utils.code_evaluation.utils import read_samples
import pandas as pd
import numpy as np

data = read_samples("evaluated_results.jsonl")
df = pd.DataFrame.from_records(data)
df["function_outputs"] = df["function_outputs"].transform(lambda x: tuple(x))

def calculate_percentage_of_correctly_parsed_asserts(df: pd.DataFrame):
    x = df.groupby('task_id')["failed_to_parse_test_cases"].aggregate(np.all).value_counts()
    return x[True] / x.sum()

def caluclate_gini_simpson_index(df: pd.DataFrame):
    per_task_id = df.groupby('task_id')



#df.groupby("task_id")[["function_outputs", "passed"]].value_counts()
#1 - s.apply(lambda x: x**2).groupby("task_id").sum()
