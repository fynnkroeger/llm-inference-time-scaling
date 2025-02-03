from typing import Any
from typing import Optional

from shared_utils.code_evaluation.utils import read_samples
from mcts.token_ids_prefix_tree import ExpectedValueSearchTreeWithDiversityPrediction, BaseTokenIdsPrefixTree
from mcts.value_estimation_network import collate_fn, NextTokenValueEstimationNN
from os import environ
import torch

from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl  # Change from pytorch_lightning to lightning.pytorch
from lightning.pytorch.loggers import CSVLogger  # Update import for logger
from random import shuffle
from value_estimator.evaluate_next_token_value_estimator import validate_model
from statistics import mean
import pandas as pd

def sample_n_from(n: int, *data: list) -> tuple:
    randomized_indicies = [i for i in range(len(data[0]))]
    shuffle(randomized_indicies)
    return tuple([data[j][i] for i in randomized_indicies[:min(n, len(data[j]))]] for j in range(len(data)))

def split_data(size_first_partition: int, *data: list) -> tuple[tuple, tuple]:
    randomized_indicies = [i for i in range(len(data[0]))]
    shuffle(randomized_indicies)
    assert size_first_partition <= len(data[0])
    return tuple([data[j][i] for i in randomized_indicies[:size_first_partition]] for j in range(len(data))), tuple([data[j][i] for i in randomized_indicies[size_first_partition:]] for j in range(len(data)))

def sample_equally_between_solved_and_unsolved(tree: BaseTokenIdsPrefixTree, n_data_points: Optional[int] = None, include_next_token_value_estimates: bool =True) -> tuple:
    X_good, y_good, metadata_good = tree.collect_value_estimator_features_and_target(discard_unsolved_problems=True, include_next_token_value_estimates=include_next_token_value_estimates)
    X_all, y_all, metadata_all = tree.collect_value_estimator_features_and_target(discard_unsolved_problems=False, include_next_token_value_estimates=include_next_token_value_estimates)
    if n_data_points is None:
        n_data_points = len(X_good)
    
    assert n_data_points <= len(X_good), f"Tried to sample {n_data_points} from X_good (max {len(X_good)})"
    
    X_good, y_good, metadata_good = sample_n_from(n_data_points, X_good, y_good, metadata_good)    
    X_all_subset, y_all_subset, metadata_all_subset = sample_n_from(n_data_points, X_all, y_all, metadata_all)
    return X_good + X_all_subset, y_good + y_all_subset, metadata_good + metadata_all_subset


if __name__ == "__main__":
    environ["CUDA_VISIBLE_DEVICES"] = "7"  # todo do this differently

    data = read_samples("outputs-samples-meta-llama-Llama-3.1-8B-t0.8.jsonl")
    original_tree = ExpectedValueSearchTreeWithDiversityPrediction.create_from_samples(data)
    tree_train, tree_val = original_tree.split(int(2/3 * 164))

    print(f"Train tree: {tree_train.number_of_unique_prompts()}. Val tree: {tree_val.number_of_unique_prompts()}")


    scores = []

    def check_data_loader(loader, name=""):
        print(f"\nChecking {name} DataLoader:")
        nan_counts = 0
        inf_counts = 0
        for batch in loader:
            # Unpack your batch structure
            inputs, value_targets, diversity_targets, solved_targets, mask = batch
            
            # Check inputs
            if torch.isnan(inputs).any():
                print(f"NaN found in inputs: {torch.isnan(inputs).sum().item()} values")
                import pdb; pdb.set_trace()
                nan_counts += torch.isnan(inputs).sum().item()
            if torch.isinf(inputs).any():
                print(f"Inf found in inputs: {torch.isinf(inputs).sum().item()} values")
                inf_counts += torch.isinf(inputs).sum().item()
            
            # Check targets
            if torch.isnan(value_targets).any():
                print(f"NaN found in value targets: {torch.isnan(value_targets).sum().item()} values")
                nan_counts += torch.isnan(value_targets).sum().item()
            if torch.isnan(diversity_targets).any():
                print(f"NaN found in diversity targets: {torch.isnan(diversity_targets).sum().item()} values")
                nan_counts += torch.isnan(diversity_targets).sum().item()
            
        return nan_counts, inf_counts

    results = []
    validation_data_size = 2000
    for n_samples in [200_000, 300_000, 400_000]:
        X_train, y_train, _ = sample_equally_between_solved_and_unsolved(tree_train, n_data_points=n_samples)
        X_val_raw, y_val_raw, metadata_val = sample_equally_between_solved_and_unsolved(tree_val)

        print(f"Train size: {len(X_train)} Val size: {len(X_val_raw)}")
        train_dataset = [(torch.Tensor(x), y[1], y[2],  y[0]) for  x, y in zip(X_train, y_train)]
        val_dataset = [(torch.Tensor(x), y[1], y[2], y[0]) for  x, y in zip(X_val_raw, y_val_raw)]


        # DataLoaders for batching
        batch_size = 512
        num_workers = 4
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

        check_data_loader(train_loader, "train")
        check_data_loader(val_loader, "val")
        model = NextTokenValueEstimationNN("meta-llama/Llama-3.1-8B", initialize_weights_with_lm_head=False)

        trainer = pl.Trainer(max_epochs=2, logger=CSVLogger("outputs/"))
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        test_results = trainer.validate(model, dataloaders=val_loader)
        print(test_results)

        mean_value_baseline = mean([y[0] for y in y_train])

        score_data = validate_model(model, val_loader, mean_value_baseline)
        # Analyze results
        print(score_data["metrics"])
        results.append(score_data["metrics"] | {
            "train_dataset_size": len(X_train),
            "validation_dataset_size": len(X_val_raw),

        })

        pd.DataFrame.from_records(results).to_csv("outputs/next_token_prediction.csv")

