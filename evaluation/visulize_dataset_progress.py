import typing
from pydantic import BaseModel, Field
import tyro
from enum import Enum
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import datetime

class XAxisScaleEnum(str, Enum):
    log = "log"
    linear = "linear"

class DatasetPlots(BaseModel):
    POSSIBLE_METRICS: typing.ClassVar[set[str]] = set(["num_tokens", "total_compute_time", "num_samples"])

    dataset_results: tyro.conf.Positional[str] = Field(description="Path to results from an dataset")
    plots_folder_name: str = Field(default_factory= lambda: datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S") + "_results")
    x_axis_scale : XAxisScaleEnum = Field(XAxisScaleEnum.log, description="Scale of the x axis")


    def calculate_solved_percent(self, data, metric):
        """Calculate percent of tasks solved over bins of a specific metric."""
        # Extract the metric values and solved statuses
        metric_values = [entry[metric] for entry in data]
        was_solved = [entry["was_solved"] for entry in data]
        
        # Sort by metric values for easier cumulative calculation
        sorted_indices = np.argsort(metric_values)
        metric_values = np.array(metric_values)[sorted_indices]
        was_solved = np.array(was_solved)[sorted_indices]
        
        # Calculate cumulative percent solved over the sorted metric values
        cumulative_solved = np.cumsum(was_solved) / len(was_solved) * 100
        dict = {metric: metric_values, "cumulative_solved": cumulative_solved}
        df = pd.DataFrame(dict)
        return df

    def extract_available_metrics(self, dataset: list[dict]) -> set[str]:
        available_metrics = set()
        for x in dataset:
            available_metrics.update(self.POSSIBLE_METRICS.intersection(set(x.keys())))
        return available_metrics
    
    def plot_graph(self, data: list[dict], metric: str):
        """Plot the percentage of tasks solved over the chosen metric."""
        df = self.calculate_solved_percent(data, metric)
        plt.figure()
        sns.lineplot(data=df, x=metric, y="cumulative_solved")
        os.makedirs(os.path.join("outputs", self.plots_folder_name), exist_ok=True)
        plt.xscale(self.x_axis_scale)
        plt.savefig(os.path.join("outputs", self.plots_folder_name, f"{metric}.png"))
            
    def execute(self):
        with open(self.dataset_results, 'r') as f:
            data = json.load(f)
        for metric in self.extract_available_metrics(data):        
            self.plot_graph(data, metric)

if  __name__ == "__main__":
    plots = tyro.cli(DatasetPlots)
    assert isinstance(plots, DatasetPlots), "tyro doesn't validate positiona list arguments correctly"
    plots.execute()