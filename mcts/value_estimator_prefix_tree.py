from email.mime import base
from typing import Optional
from mcts.token_ids_prefix_tree import BaseTokenIdsPrefixTree, TokenIdNode
import math

# Value estimation deps
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from random import shuffle
from mcts.value_estimation_network import MultiLayerNN, NextTokenValueEstimationNN, collate_fn, NextTokenValueAdapterNN
from tqdm import tqdm
import pandas as pd
from statistics import variance

PREDICTION_BATCH_SIZE = 512
TRAIN_BATCH_SIZE = 512

DATALOADER_NUM_WORKERS = 1

LLAMA_1_B_HIDDEN_STATES_SIZE = 2048

OUTPUT_DIR = "outputs/next_token_value_estimator"
from os import environ

def sample_n_from(n: int, *data: list) -> tuple:
    randomized_indicies = [i for i in range(len(data[0]))]
    shuffle(randomized_indicies)
    return tuple([data[j][i] for i in randomized_indicies[:min(n, len(data[j]))]] for j in range(len(data)))

def split_data(size_first_partition: int, *data: list) -> tuple[tuple, tuple]:
    randomized_indicies = [i for i in range(len(data[0]))]
    shuffle(randomized_indicies)
    assert size_first_partition <= len(data[0])
    return tuple([data[j][i] for i in randomized_indicies[:size_first_partition]] for j in range(len(data))), tuple([data[j][i] for i in randomized_indicies[size_first_partition:]] for j in range(len(data)))

class ValueEstimatorTokenIdsPrefixTree(BaseTokenIdsPrefixTree):
    value_estimation_model: pl.LightningModule | None
    node_value_estimation_queue: set[TokenIdNode]

    def __init__(self) -> None:
        super().__init__()
        self.node_value_estimation_queue = set()
        self.model_iteration = -1
        self.value_estimation_model = None
        self.logs = []

    def log(self, data: dict) -> None:
        self.logs.append(data | {"iteration": self.iteration, "model_iteration": self.model_iteration})
    
    def flush_logs(self) -> None:
        if len(self.logs) > 0:
            pd.DataFrame.from_records(self.logs).to_csv(f"{OUTPUT_DIR}/value_estimator_prefix_tree_logs-{self.iteration}.csv")
            self.logs = []

    def _additional_node_attributes(self, token_id: Optional[int], node_log_prob: float) -> dict:
        return {
            "total_following_paths_probability": 0.0,
            "following_paths_value_estimate": 1.0,
            "following_paths_value_estimate_model_iteration": -2 # Here we track from which iteration our estimate is. We may throw away old estimates
        }
    
    def _update_node_metrics(self, node: TokenIdNode, continuation_probability: float, path_length: int, hashed_function_outputs: Optional[int]) -> None:
        """Update node's expected value upper bound by adding the continuation probability."""
        node["total_following_paths_probability"] += continuation_probability

        # Compute estimate in batch lazily
        # iteartion is increased after adding all sequences
        will_train_new_model_in_next_iteration = self.will_train_new_model(self.iteration + 1)
        if isinstance(node, TokenIdNode):
            if (will_train_new_model_in_next_iteration or node["following_paths_value_estimate_model_iteration"] < self.model_iteration) and node not in self.node_value_estimation_queue:
                if node.value_estimator_features() is not None:
                    self.node_value_estimation_queue.add(node)
        else:
            print("Unkown object added:", node)
        
    def _get_adjustment_factor(self, node: TokenIdNode) -> float:
        """
        Get adjustment factor based on expected value upper bound.
        As we explore more failing paths, the upper bound decreases.
        """
        
        """
        if node["following_paths_value_estimate"] < 0.2:
            numerically_stable_value_penalty = 1.0 - node["following_paths_value_estimate"]
        else:
            numerically_stable_value_penalty = 0.0
        """
        # Decay (increase the probability to 1 to increase again)
        # node["following_paths_value_estimate"] = (1.0 - node["following_paths_value_estimate"]) * 0.05 * (self.iteration - node["following_paths_value_estimate_model_iteration"]) + node["following_paths_value_estimate"]
        #return math.log(1.0 - max(node["total_following_paths_probability"], numerically_stable_value_penalty)) if node["total_following_paths_probability"] < 1.0 else -math.inf
        if node["following_paths_value_estimate"] <= 0.0:
            return -math.inf
        elif node["following_paths_value_estimate"] >= 1.0:
            return 0.0
        else:
            return math.log(node["total_following_paths_probability"])
        
        #return math.log(1.0 - node["total_following_paths_probability"]) if node["total_following_paths_probability"] < 1.0 else -math.inf
    def will_train_new_model(self, iteration: int) -> bool:
        return iteration % 5 == 0
    
    def finished_adding_sequences_watermark(self) -> None:
        super().finished_adding_sequences_watermark()
        # Only train a new model in iteration 1, 2, 4, 8, ... 2**i
        if self.will_train_new_model(self.iteration):
            self._train_value_estimation_model()
        if self.value_estimation_model is not None:
            self._batch_compute_value_estimates()
        self.flush_logs()


    def _train_value_estimation_model(self) -> None:
        # Data in X_good is mostly 1.0 estimates. In all we mostly we have 0.0 values. Use all from good and sample equally from all 
        # This avoids models that "converge" by only predicting all 0 or all 1
        X_good, y_good, _ = self.collect_value_estimator_features_and_target(discard_unsolved_problems=True)
        X_all, y_all, _  = self.collect_value_estimator_features_and_target(discard_unsolved_problems=False)
        X_all_subset, y_all_subset = sample_n_from(len(X_good), X_all, y_all)
        X, y = torch.Tensor(X_good + X_all_subset), torch.Tensor(y_good + y_all_subset)
        print(f"Using {len(X)} datapoints to train in iteration {self.iteration}")
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=DATALOADER_NUM_WORKERS)
        
        # Hidden_sizes=[] means only one single layer
        model = MultiLayerNN(LLAMA_1_B_HIDDEN_STATES_SIZE, hidden_sizes=[])
        trainer = pl.Trainer(max_epochs=25, accelerator="cpu")
        trainer.fit(model, train_dataloaders=train_loader)
        self.value_estimation_model = model
        self.model_iteration = self.iteration


    """
    Estimates node values from all nodes in the node_value_estimation_queue
    """ 
    def _batch_compute_value_estimates(self) -> None:
        assert self.value_estimation_model is not None

        # Cast to list. I am not sure if python set always has the same predifined order and we want access via an index
        node_value_estimation_list = list(self.node_value_estimation_queue)
        X = torch.Tensor([node.value_estimator_features() for node in node_value_estimation_list])
        dataset = TensorDataset(X)
        prediction_dataloader = DataLoader(dataset, PREDICTION_BATCH_SIZE, shuffle=False, num_workers=DATALOADER_NUM_WORKERS)

        self.value_estimation_model.eval()
        value_estimates = []
        with torch.inference_mode():
            for (X_batch,) in tqdm(prediction_dataloader, "Estimating values from hidden_states"):
                preds = self.value_estimation_model(X_batch).squeeze().tolist()
                if isinstance(preds, float): # Prediction on a single value. Not sure why it does not return a list
                    value_estimates.append(preds)
                else:
                    for y in preds:
                        value_estimates.append(y)
        
        for node, value_estimate in zip(node_value_estimation_list, value_estimates):
            node["following_paths_value_estimate"] = value_estimate
            node["following_paths_value_estimate_iteration"] = self.iteration

        # Clean up queue so we dont process them twice
        self.node_value_estimation_queue = set()

    def _get_advantage_adjusted_logits(
        self, 
        prompt_token_ids: list[int], 
        output_token_ids: list[int], 
        logits: torch.Tensor, 
        max_prob_shift_percent: float = 0.2  # Hyperparameter for max probability change
    ) -> torch.Tensor:
        # Traverse to the current node
        prompt_token_ids_as_tuple = tuple(prompt_token_ids)
        if prompt_token_ids_as_tuple not in self.prompt_root:
            return logits
        
        node = self.prompt_root[prompt_token_ids_as_tuple]
        path_depth = 0
        for token_id in output_token_ids:
            if token_id not in node["children_token_ids"]:
                return logits
            node = node["children_token_ids"][token_id]
            path_depth += 1
        
        # Get child token IDs and value estimates
        child_token_ids = list(node["children_token_ids"].keys())
        child_value_estimates = [
            node["children_token_ids"][tid]["following_paths_value_estimate"] 
            for tid in child_token_ids
        ]
        
        # Skip if insufficient children
        if len(child_token_ids) <= 1:
            return logits
        
        # Compute baseline and relative advantages
        baseline_value = sum(child_value_estimates) / len(child_value_estimates)

        mean_logit_size = torch.abs(logits).mean()
        mean_logit_size_float = mean_logit_size.item()
        without_abs_mean_logit_size = logits.mean().item()
        # Scale adjustments based on original logits and value estimates
        adjustments = []
        for token_id, val_estimate in zip(child_token_ids, child_value_estimates):
            # Compute difference from baseline

            advantage = val_estimate - baseline_value
    
            # Ensure we don't exceed max_prob_shift_percent
            adjustment: torch.Tensor = advantage * torch.abs(mean_logit_size)
            adjustments.append(adjustment)
            self.log({
                "advantage": advantage,
                "value_estimate": val_estimate,
                "baseline_value": baseline_value,
                "logit_adjustment": adjustment.item(),
                "original_logit": logits[token_id].item(),
                "mean_logit": mean_logit_size_float,
                "non_abs_mean_logit": without_abs_mean_logit_size,
                "path_depth": path_depth,
                "token_id": token_id,
                "num_adjusted_tokens": len(child_value_estimates)
            })

        # Convert to tensor
        adjustments_tensor = torch.tensor(adjustments, device=logits.device)
        child_token_ids_tensor = torch.tensor(child_token_ids, device=logits.device, dtype=torch.long)
        
        # Adjust logits
        logits[child_token_ids_tensor] += adjustments_tensor
        
        return logits
    
    def _get_parent_advantage_adjusted_logits(
        self, 
        prompt_token_ids: list[int], 
        output_token_ids: list[int], 
        logits: torch.Tensor, 
        max_prob_shift_percent: float = 0.2  # Hyperparameter for max probability change
    ) -> torch.Tensor:
        # Traverse to the current node
        prompt_token_ids_as_tuple = tuple(prompt_token_ids)
        if prompt_token_ids_as_tuple not in self.prompt_root:
            return logits
        
        node = self.prompt_root[prompt_token_ids_as_tuple]
        path_depth = 0
        for token_id in output_token_ids:
            if token_id not in node["children_token_ids"]:
                return logits
            node = node["children_token_ids"][token_id]
            path_depth += 1
        
        if node["parent"] is None:
            return logits
        
        # Get child token IDs and value estimates
        child_token_ids = list(node["children_token_ids"].keys())
        child_value_estimates = [
            node["children_token_ids"][tid]["following_paths_value_estimate"] 
            for tid in child_token_ids
        ]
        mean_logit_size = torch.abs(logits).mean()
        
        # Skip if insufficient children
        if len(child_token_ids) <= 1:
            return logits
        
        # Compute baseline and relative advantages
        baseline_value = node["parent"]["following_paths_value_estimate"]

      
        # Scale adjustments based on original logits and value estimates
        adjustments = []
        for token_id, val_estimate in zip(child_token_ids, child_value_estimates):
            # Compute difference from baseline

            advantage = val_estimate - baseline_value
    
            # Ensure we don't exceed max_prob_shift_percent
            adjustment: torch.Tensor = advantage * torch.abs(mean_logit_size)
            adjustments.append(adjustment)
            self.log({
                "advantage": advantage,
                "value_estimate": val_estimate,
                "baseline_value": baseline_value,
                "logit_adjustment": adjustment.item(),
                "original_logit": logits[token_id].item(),
                "path_depth": path_depth,
                "token_id": token_id,
                "num_adjusted_tokens": len(child_value_estimates)
            })

        # Convert to tensor
        adjustments_tensor = torch.tensor(adjustments, device=logits.device)
        child_token_ids_tensor = torch.tensor(child_token_ids, device=logits.device, dtype=torch.long)
        
        # Adjust logits
        logits[child_token_ids_tensor] += adjustments_tensor
        
        return logits
    

class NextTokenValueEstimatorTokenIdsPrefixTree(ValueEstimatorTokenIdsPrefixTree):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id = model_id

    def _additional_node_attributes(self, token_id: Optional[int], node_log_prob: float) -> dict:
        return {
            "total_following_paths_probability": 0.0, # Not used but initlialized to avoid crashes
            "estimated_next_token_values": None,
            "following_paths_value_estimate_model_iteration": -2 # Here we track from which iteration our estimate is. We may throw away old estimates
        }
    
    def _batch_compute_value_estimates(self) -> None:
        assert self.value_estimation_model is not None

        # Cast to list. I am not sure if python set always has the same predifined order and we want access via an index
        node_value_estimation_list = list(self.node_value_estimation_queue)
        X = torch.Tensor([node.value_estimator_features() for node in node_value_estimation_list])
        dataset = TensorDataset(X)
        prediction_dataloader = DataLoader(dataset, PREDICTION_BATCH_SIZE, shuffle=False, num_workers=DATALOADER_NUM_WORKERS)

        self.value_estimation_model.eval()
        value_estimates = []
        with torch.inference_mode():
            for (X_batch,) in tqdm(prediction_dataloader, f"Estimating {len(node_value_estimation_list)} values from hidden_states"):
                preds = self.value_estimation_model(X_batch).squeeze()
                for batch_idx in range(X_batch.size(0)):
                    value_estimates.append(preds[batch_idx])
        
        for node, value_estimate in zip(node_value_estimation_list, value_estimates):
            node["estimated_next_token_values"] = value_estimate
            node["following_paths_value_estimate_iteration"] = self.iteration

        # Clean up queue so we dont process them twice
        self.node_value_estimation_queue = set()

    def _train_value_estimation_model(self) -> None:
        # Data in X_good is mostly 1.0 estimates. In all we mostly we have 0.0 values. Use all from good and sample equally from all 
        # This avoids models that "converge" by only predicting all 0 or all 1
        X_good, y_good, _ = self.collect_value_estimator_features_and_target(discard_unsolved_problems=True, include_next_token_value_estimates=True)
        X_all, y_all, _  = self.collect_value_estimator_features_and_target(discard_unsolved_problems=False, include_next_token_value_estimates=True)
        X_all_subset, y_all_subset = sample_n_from(len(X_good), X_all, y_all, )
        X, y = torch.Tensor(X_good + X_all_subset), y_good + y_all_subset
        print(f"Using {len(X)} datapoints to train in iteration {self.iteration}")
        dataset = [(torch.Tensor(x), y[1], y[2],  y[0]) for  x, y in zip(X, y)]
        train_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=DATALOADER_NUM_WORKERS, collate_fn=collate_fn) # type: ignore
        
        # Hidden_sizes=[] means only one single layer
        model = NextTokenValueEstimationNN(self.model_id, device="cpu")
        trainer = pl.Trainer(max_epochs=25, accelerator="cpu")
        trainer.fit(model, train_dataloaders=train_loader)
        self.value_estimation_model = model
        self.model_iteration = self.iteration

    def _get_advantage_adjusted_logits(
        self, 
        prompt_token_ids: list[int], 
        output_token_ids: list[int], 
        logits: torch.Tensor, 
        max_prob_shift_percent: float = 0.2  # Hyperparameter for max probability change
    ) -> torch.Tensor:
        # Traverse to the current node
        prompt_token_ids_as_tuple = tuple(prompt_token_ids)
        if prompt_token_ids_as_tuple not in self.prompt_root:
            return logits
        
        node = self.prompt_root[prompt_token_ids_as_tuple]
        path_depth = 0
        for token_id in output_token_ids:
            if token_id not in node["children_token_ids"]:
                self.log({
                    "has_cached_estimated_values": False,
                    "is_on_new_path": True,
                    "path_depth": path_depth
                })
                return logits
            node = node["children_token_ids"][token_id]
            path_depth += 1
        
        next_token_values: torch.Tensor | None = node["estimated_next_token_values"]
        if next_token_values is None:
            self.log({
                "has_cached_estimated_values": False,
                "is_on_new_path": False,
                "path_depth": path_depth
            })
            return logits

        next_token_values = next_token_values.to(logits.device)
        # Compute baseline and relative advantages
        baseline_value = next_token_values.mean()
        advantage = next_token_values - baseline_value
        logit_adjustment = logits.abs() * advantage
        logits = logits + logit_adjustment

        self.log({
                "has_cached_estimated_values": True,
                "is_on_new_path": False,
                "mean_value": baseline_value.item(),
                "var_value": next_token_values.var().item(),
                "mean_logit_adjustment": logit_adjustment.mean().item(),
                "var_logit_adjustment": logit_adjustment.var().item(),
                "path_depth": path_depth
            })
        
        return logits

    def will_train_new_model(self, iteration: int) -> bool:
        return iteration >= 20 and iteration % 10 == 0
    
class NextTokenAdvantageAdapterTokenIdsPrefixTree(NextTokenValueEstimatorTokenIdsPrefixTree):
    def _additional_node_attributes(self, token_id: Optional[int], node_log_prob: float) -> dict:
        return {
            "total_following_paths_probability": 0.0, # Not used but initlialized to avoid crashes
            "estimated_next_token_values": None,
            "following_paths_value_estimate_model_iteration": -2 # Here we track from which iteration our estimate is. We may throw away old estimates
        }
    
    def _batch_compute_value_estimates(self) -> None:
        assert self.value_estimation_model is not None

        # Cast to list. I am not sure if python set always has the same predifined order and we want access via an index
        node_value_estimation_list = list(self.node_value_estimation_queue)
        X = torch.Tensor([node.value_estimator_features() for node in node_value_estimation_list])
        dataset = TensorDataset(X)
        prediction_dataloader = DataLoader(dataset, PREDICTION_BATCH_SIZE, shuffle=False, num_workers=DATALOADER_NUM_WORKERS)

        self.value_estimation_model.eval()
        value_estimates = []
        with torch.inference_mode():
            for (X_batch,) in tqdm(prediction_dataloader, f"Estimating {len(node_value_estimation_list)} values from hidden_states"):
                preds = self.value_estimation_model(X_batch).squeeze()
                for batch_idx in range(X_batch.size(0)):
                    value_estimates.append(preds[batch_idx])
        
        for node, value_estimate in zip(node_value_estimation_list, value_estimates):
            node["estimated_next_token_values"] = value_estimate
            node["following_paths_value_estimate_iteration"] = self.iteration

        # Clean up queue so we dont process them twice
        self.node_value_estimation_queue = set()

    def _train_value_estimation_model(self) -> None:
        # Data in X_good is mostly 1.0 estimates. In all we mostly we have 0.0 values. Use all from good and sample equally from all 
        # This avoids models that "converge" by only predicting all 0 or all 1
        X_good, y_good, _ = self.collect_value_estimator_features_and_target(discard_unsolved_problems=True, include_next_token_value_estimates=True)
        X_all, y_all, _  = self.collect_value_estimator_features_and_target(discard_unsolved_problems=False, include_next_token_value_estimates=True)
        X_all_subset, y_all_subset = sample_n_from(len(X_good), X_all, y_all, )
        X, y = torch.Tensor(X_good + X_all_subset), y_good + y_all_subset
        print(f"Using {len(X)} datapoints to train in iteration {self.iteration}")
        dataset = [(torch.Tensor(x), y[1], y[2],  y[0]) for  x, y in zip(X, y)]
        train_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=DATALOADER_NUM_WORKERS, collate_fn=collate_fn) # type: ignore
        
        # Hidden_sizes=[] means only one single layer
        model = NextTokenValueEstimationNN(self.model_id, device="cpu")
        trainer = pl.Trainer(max_epochs=25, accelerator="cpu")
        trainer.fit(model, train_dataloaders=train_loader)
        self.value_estimation_model = model
        self.model_iteration = self.iteration

    def _get_adjusted_logits_fast(
        self, 
        prompt_token_ids: list[int], 
        output_token_ids: list[int], 
        logits: torch.Tensor, 
    ) -> torch.Tensor:
        # Traverse to the current node
        prompt_token_ids_as_tuple = tuple(prompt_token_ids)
        if prompt_token_ids_as_tuple not in self.prompt_root:
            return logits
        
        node = self.prompt_root[prompt_token_ids_as_tuple]
        path_depth = 0
        for token_id in output_token_ids:
            if token_id not in node["children_token_ids"]:
                self.log({
                    "has_cached_estimated_values": False,
                    "is_on_new_path": True,
                    "path_depth": path_depth
                })
                return logits
            node = node["children_token_ids"][token_id]
            path_depth += 1
        
        next_token_values: torch.Tensor | None = node["estimated_next_token_values"]
        if next_token_values is None:
            self.log({
                "has_cached_estimated_values": False,
                "is_on_new_path": False,
                "path_depth": path_depth
            })
            return logits

        next_token_values = next_token_values.to(logits.device)
        # Compute baseline and relative advantages
        baseline_value = next_token_values.mean()
        advantage = next_token_values - baseline_value
        logit_adjustment = logits.abs() * advantage
        logits = logits + logit_adjustment

        self.log({
                "has_cached_estimated_values": True,
                "is_on_new_path": False,
                "mean_value": baseline_value.item(),
                "var_value": next_token_values.var().item(),
                "mean_logit_adjustment": logit_adjustment.mean().item(),
                "var_logit_adjustment": logit_adjustment.var().item(),
                "path_depth": path_depth
            })
        
        return logits
