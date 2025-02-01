from typing import Any

from pydantic import conint
from tqdm import tqdm

from shared_utils.code_evaluation.utils import read_samples
from mcts.token_ids_prefix_tree import ExpectedValueSearchTreeWithDiversityPrediction
from os import environ
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import lightning.pytorch as pl  # Change from pytorch_lightning to lightning.pytorch
from lightning.pytorch.loggers import CSVLogger  # Update import for logger
from random import shuffle


environ["CUDA_VISIBLE_DEVICES"] = ""  # todo do this differently

data = read_samples("outputs/samples-t0.8-with-hidden-states.jsonl")

tree = ExpectedValueSearchTreeWithDiversityPrediction.create_from_samples(data)
X_good, y_good, metadata_good = tree.collect_value_estimator_features_and_target(discard_unsolved_problems=True)
X_all, y_all, metadata_all = tree.collect_value_estimator_features_and_target(discard_unsolved_problems=False)

randomized_indicies = [i for i in range(len(X_all))]
shuffle(randomized_indicies)

def sample_n_from(n: int, *data: list) -> tuple:
    randomized_indicies = [i for i in range(len(data[0]))]
    shuffle(randomized_indicies)
    return tuple([data[j][i] for i in randomized_indicies[:min(n, len(data[j]))]] for j in range(len(data)))

def split_data(size_first_partition: int, *data: list) -> tuple[tuple, tuple]:
    randomized_indicies = [i for i in range(len(data[0]))]
    shuffle(randomized_indicies)
    assert size_first_partition <= len(data[0])
    return tuple([data[j][i] for i in randomized_indicies[:size_first_partition]] for j in range(len(data))), tuple([data[j][i] for i in randomized_indicies[size_first_partition:]] for j in range(len(data)))

results = []
for n_samples in [2**i for i in range(5, 9)]:

    n_training_points = n_samples * 164
    X_good_subset, y_good_subset, metadata_good_subset = sample_n_from(n_training_points, X_good, y_good, metadata_good)
    X_all_subset, y_all_subset, metadata_all_subset = sample_n_from(n_training_points, X_all, y_all, metadata_all)

    X, y = X_good_subset + X_all_subset, y_good_subset + y_all_subset
    metadata = metadata_good_subset + metadata_all_subset

    train_size = int(0.8 * len(X))
    (X_train, y_train, _), (X_val_raw, y_val_raw, metadata_val) = split_data(train_size, X, y, metadata)
    print(f"Train size: {len(X_train)} Val size: {len(X_val_raw)}")
    
    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    X_val, y_val = torch.Tensor(X_val_raw), torch.Tensor(y_val_raw)


    # Split into training and validation datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)


    # Define the Lightning Module
    class MultiLayerNN(pl.LightningModule):
        def __init__(self, input_size, hidden_sizes=None, learning_rate=0.001, use_bce_loss=False):
            super(MultiLayerNN, self).__init__()
            if hidden_sizes is None:
                hidden_sizes = [1028, 512, 256, 128]  # Default hidden layer sizes

            layers = []
            previous_size = input_size

            # Adding hidden layers with ReLU activation
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(previous_size, hidden_size))
                layers.append(nn.ReLU())
                previous_size = hidden_size

            # Adding the final layer
            layers.append(nn.Linear(previous_size, 1))  # Output layer with a single unit
            layers.append(nn.Sigmoid())  # Sigmoid activation for probabilities

            self.model = nn.Sequential(*layers)
            self.learning_rate = learning_rate
            self.criterion = nn.BCELoss() if use_bce_loss else nn.MSELoss(reduction="none")
            self.save_hyperparameters()

        def forward(self, x):
            return self.model(x)

        
        def training_step(self, batch, batch_idx):
            X, y = batch
            y = y.view(-1, 1)  # Reshape y to [batch_size, 1]
            outputs = self(X)
            loss = self.criterion(outputs, y).mean()
            self.log('train_loss', loss, prog_bar=True)  # Log the training loss
            return loss

        def validation_step(self, batch, batch_idx):
            X, y = batch
            y = y.view(-1, 1)  # Reshape y to [batch_size, 1]
            outputs = self(X)
            loss = self.criterion(outputs, y).mean()
            self.log('val_loss', loss, prog_bar=True)  # Log the validation loss
            return loss

        def configure_optimizers(self):
            return optim.Adam(self.parameters(), lr=self.learning_rate)

    input_size = 2048
    # DataLoaders for batching
    batch_size = 512
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    # Initialize the model
    model = MultiLayerNN(input_size, [])

    # Train the model
    trainer = pl.Trainer(max_epochs=25, logger=CSVLogger("outputs/"))
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    test_results = trainer.validate(model, dataloaders=val_loader)
    print(test_results)

    import math
    def predict_value_estimates(model, val_loader: DataLoader):
        model.eval()  # Set the model to evaluation mode
        predictions = []
        with torch.no_grad():
            for X_val, targets in val_loader:
                preds = model(X_val).squeeze().tolist()
                for y, y_star_torch in zip(preds, targets):
                    y_star = y_star_torch.item()

                    error = abs(y - y_star)

                    predictions.append({
                        "value_estimates": y,
                        "error": error,
                        "values": y_star,
                        "is_promising_path": y_star > 0.0
                    })
                
        return predictions

    predictions = predict_value_estimates(model, val_loader)
    import seaborn as sns
    # Plot the predicted values as a histogram/distplot

    import pandas as pd
    df = pd.DataFrame.from_records(predictions)
    plt = sns.histplot(data=df, x="values", hue="is_promising_path", kde=False, bins=30, ); plt.figure.savefig("value_estimates_validation_set.png")

    scores_random_baseline = []
    scores_value_estimator = []

    # Split validation by the prompt hash (should be unique the same way task id is. But we dont fit to our datastructure of a dataset with task ids)
    import random
    from collections import defaultdict
    val_data_by_prompt: dict[int, tuple[list, list, list]] = defaultdict(lambda: ([], [], []))

    # is not index. Just wanted to avoid overwriting old variables
    for x_i, y_i, m_i in zip(X_val_raw, y_val_raw, metadata_val):
        val_data_by_prompt[m_i["prompt_hash"]][0].append(x_i)
        val_data_by_prompt[m_i["prompt_hash"]][1].append(y_i)
        val_data_by_prompt[m_i["prompt_hash"]][2].append(m_i)

    val_data_per_task: list[tuple[torch.Tensor, list[float], list[dict]]] = []
    for k, (x_i, y_i, m_i) in val_data_by_prompt.items():
        val_data_per_task.append((torch.Tensor(x_i), y_i, m_i))
    
    from statistics import mean, variance

    score_data = []
    model.eval()
    from itertools import combinations
    with torch.no_grad():
        for (X_val, y_targets, m_i) in tqdm(val_data_per_task, "Evaluting per task id decider performance"):
            preds = model(X_val).squeeze().tolist()
            if len(y_targets) < 2:
                continue
            for i, j in combinations(range(len(y_targets)), 2):
                if i == j or y_targets[i] == y_targets[j]:
                    continue
                
                path_depth_difference = abs(m_i[i]["path_depth"] - m_i[j]["path_depth"])
                mean_path_depth = (m_i[i]["path_depth"] + m_i[j]["path_depth"]) / 2 
                x_1 = preds[i]
                x_2 = preds[j]
                
                score = None
                if y_targets[i] > y_targets[j]:
                    scores_random_baseline.append(1)
                    if x_1 > x_2:
                        score = 1
                    else:
                        score = 0
                else:
                    scores_random_baseline.append(0)
                    if x_1 < x_2:
                        score = 1
                    else:
                        score = 0

                score_data.append({
                    "score": score,
                    "mean_path_depth": mean_path_depth,
                    "depth_1": m_i[i]["path_depth"],
                    "depth_2": m_i[j]["path_depth"],
                    "path_depth_difference": path_depth_difference
                })
                scores_value_estimator.append(score)

    df_score = pd.DataFrame.from_records(score_data)
    import matplotlib.pyplot as plt
    df_score.to_csv(f"scores_df-{len(X_train)}.csv")
    plt.clf()
    sns.barplot(df_score, x="path_depth_difference", y="score").figure.savefig(f"decider_by_height-{len(X_train)}.png")
    
    print("Per example average")
    print(mean(scores_random_baseline), variance(scores_random_baseline))
    print(mean(scores_value_estimator), variance(scores_value_estimator), len(scores_value_estimator))
    scores_random_baseline = []
    scores_value_estimator = []

    for _ in range(5000):
        i, j = 0,0
        while i == j:
            i, j = random.randrange(0, len(predictions) - 1), random.randrange(0, len(predictions) - 1)
        x_1 = predictions[i]
        x_2 = predictions[j]
        if x_1["values"] == x_2["values"]:
            continue
        if x_1["values"] > x_2["values"]:
            scores_random_baseline.append(1)
            if x_1["value_estimates"] > x_2["value_estimates"]:
                scores_value_estimator.append(1)
            else:
                scores_value_estimator.append(0)
        else:
            scores_random_baseline.append(0)
            if x_1["value_estimates"] < x_2["value_estimates"]:
                scores_value_estimator.append(1)
            else:
                scores_value_estimator.append(0)


    print(n_samples)
    print(mean(scores_random_baseline), variance(scores_random_baseline))
    print(mean(scores_value_estimator), variance(scores_value_estimator))

    results.append({
        "num_samples_per_problem": n_samples,
        "n_training_set": len(train_dataset),
        "n_validation_set": len(val_dataset),
        "model_value_decider_mean": mean(scores_value_estimator),
        "model_value_decider_var": variance(scores_value_estimator),
        "random_value_decider_mean": mean(scores_random_baseline),
        "random_value_decider_var": variance(scores_random_baseline)

    })
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df_results = pd.DataFrame.from_records(results)
plt.clf(); p = sns.lineplot(df_results, x="num_samples_per_problem", y="model_value_decider_mean"); p.set_xscale('linear'); p.figure.savefig("results_2.png")