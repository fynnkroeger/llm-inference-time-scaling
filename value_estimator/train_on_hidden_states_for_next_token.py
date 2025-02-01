from typing import Any

from typing import Optional
from tqdm import tqdm

from shared_utils.code_evaluation.utils import read_samples
from mcts.token_ids_prefix_tree import ExpectedValueSearchTreeWithDiversityPrediction, BaseTokenIdsPrefixTree
from mcts.value_estimation_network import collate_fn, NextTokenValueEstimationNN
from os import environ
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import lightning.pytorch as pl  # Change from pytorch_lightning to lightning.pytorch
from lightning.pytorch.loggers import CSVLogger  # Update import for logger
from random import shuffle
from value_estimator.evaluate_next_token_value_estimator import evaluate_per_prompt, validate_model
from collections import defaultdict
from statistics import mean, variance, stdev
import pandas as pd

environ["CUDA_VISIBLE_DEVICES"] = "7"  # todo do this differently


data = read_samples("outputs/samples-t0.8-with-hidden-states.jsonl")

tree = ExpectedValueSearchTreeWithDiversityPrediction.create_from_samples(data)
tree_train, tree_val = tree.split(int(2/3 * 164))
print(f"Train tree: {tree_train.number_of_unique_prompts()}. Val tree: {tree_val.number_of_unique_prompts()}")

def sample_equally_between_solved_and_unsolved(tree: BaseTokenIdsPrefixTree, n_data_points: Optional[int] = None) -> tuple:
    X_good, y_good, metadata_good = tree.collect_value_estimator_features_and_target(discard_unsolved_problems=True, include_next_token_value_estimates=True)
    X_all, y_all, metadata_all = tree.collect_value_estimator_features_and_target(discard_unsolved_problems=False, include_next_token_value_estimates=True)
    if n_data_points is None:
        n_data_points = len(X_good)
    else:
         X_good, y_good, metadata_good = sample_n_from(n_data_points, X_good, y_good, metadata_good)
    
    X_all_subset, y_all_subset, metadata_all_subset = sample_n_from(n_data_points, X_all, y_all, metadata_all)
    return X_good + X_all_subset, y_good + y_all_subset, metadata_good + metadata_all_subset


scores = []
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
print(f"Total available data: {len(X_all)}. Total available good (value>0.0) data: {len(X_good)}")
validation_data_size = 2000
for n_samples in [2**i for i in range(0, 9)]:

    n_training_points = n_samples * 164
    total_points = n_training_points + validation_data_size
    X_good_subset, y_good_subset, metadata_good_subset = sample_n_from(total_points, X_good, y_good, metadata_good)
    X_all_subset, y_all_subset, metadata_all_subset = sample_n_from(total_points * 4, X_all, y_all, metadata_all)

    X, y = X_good_subset + X_all_subset, y_good_subset + y_all_subset
    metadata = metadata_good_subset + metadata_all_subset

    (X_train, y_train, _), (X_val_raw, y_val_raw, metadata_val) = split_data(n_training_points, X, y, metadata)
    print(f"Train size: {len(X_train)} Val size: {len(X_val_raw)}")
    
    train_dataset = [(torch.Tensor(x), y[1], y[2],  y[0]) for  x, y in zip(X_train, y_train)]
    val_dataset = [(torch.Tensor(x), y[1], y[2], y[0]) for  x, y in zip(X_val_raw, y_val_raw)]


    # DataLoaders for batching
    batch_size = 512
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    model = NextTokenValueEstimationNN("meta-llama/Llama-3.2-1B", initialize_weights_with_lm_head=False)

    trainer = pl.Trainer(max_epochs=25, logger=CSVLogger("outputs/"))
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    test_results = trainer.validate(model, dataloaders=val_loader)
    print(test_results)

    score_data = validate_model(model, val_loader)
    # Analyze results
    print(score_data["metrics"])
    results.append(score_data["metrics"])

    pd.DataFrame.from_records(results).to_csv("next_token_prediction.csv")

