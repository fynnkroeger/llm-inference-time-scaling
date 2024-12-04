from typing import Self, TypedDict, Union, Optional, NotRequired
from abc import ABC, abstractmethod
import math
import torch

class TokenIdNode(TypedDict):
    token_id: Optional[int]  # is None for root
    node_log_prob: float
    total_following_paths_probability: NotRequired[float]  # Sum of probabilities from this node to end for all explored paths
    avg_continuation_probability: NotRequired[float]  # Average probability from this node to end
    path_count: NotRequired[int]    # Number of paths through this node
    children_token_ids: dict[int, Self]

class BaseTokenIdsPrefixTree(ABC):
    def __init__(self) -> None:
        self.prompt_root: dict[tuple[int,...], TokenIdNode] = {}

    def _create_empty_node(self, token_id: Optional[int], node_log_prob: float) -> TokenIdNode:
        return {
            "token_id": token_id,
            "node_log_prob": node_log_prob,
            "children_token_ids": {}
        } | self._additional_node_attributes(token_id, node_log_prob) # type: ignore
    
    
    def _additional_node_attributes(self, token_id: Optional[int], node_log_prob: float) -> dict:
        return {

        }

    @abstractmethod
    def _update_node_metrics(self, node: TokenIdNode, continuation_probability: float, path_length: int, hashed_function_outputs:Optional[int]) -> None:
        """Update the node's metrics based on the specific search strategy."""
        pass

    @abstractmethod
    def _get_adjustment_factor(self, node: TokenIdNode) -> float:
        """Calculate the adjustment factor for logits based on the specific search strategy."""
        pass

    def add_sequence(self, prompt_token_ids: list[int], token_ids: list[int], log_probs: list[float], hashed_function_outputs : Optional[int]=None) -> None:
        assert len(token_ids) == len(log_probs), "Each token_id must have one log_prob. However, the two lists have different lengths"
        
        # Calculate continuation probabilities from each position to the end
        continuation_log_probs = [0.0] * len(log_probs)
        cumulative_log_prob = 0.0
        for i in range(len(log_probs) - 1, -1, -1):
            cumulative_log_prob += log_probs[i]
            continuation_log_probs[i] = cumulative_log_prob

        prompt_token_ids_as_tuple = tuple(prompt_token_ids)
        if prompt_token_ids_as_tuple not in self.prompt_root:
            self.prompt_root[prompt_token_ids_as_tuple] = self._create_empty_node(None, 0.0)

        node = self.prompt_root[prompt_token_ids_as_tuple]
        for i in range(len(token_ids)):
            continuation_probability = math.exp(continuation_log_probs[i])
            self._update_node_metrics(node, continuation_probability, len(token_ids) - i, hashed_function_outputs)
            if token_ids[i] not in node["children_token_ids"]:
                node["children_token_ids"][token_ids[i]] = self._create_empty_node(token_ids[i], log_probs[i])
            node = node["children_token_ids"][token_ids[i]]

    def adjust_logits_fast(self, prompt_token_ids: list[int], output_token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        prompt_token_ids_as_tuple = tuple(prompt_token_ids)

        if prompt_token_ids_as_tuple not in self.prompt_root:
            return logits

        node = self.prompt_root[prompt_token_ids_as_tuple]
        for token_id in output_token_ids:
            if token_id not in node["children_token_ids"]:
                return logits
            node = node["children_token_ids"][token_id]

        adjusted_token_ids = list(node["children_token_ids"].keys())
        adjustment_factors = torch.tensor(
            [self._get_adjustment_factor(x) for x in node["children_token_ids"].values()],
            device=logits.device
        )

        adjusted_token_ids_tensor = torch.tensor(adjusted_token_ids, device=logits.device, dtype=torch.long)

        exp_logits = torch.exp(logits)
        mask_adjusted = torch.zeros_like(logits, dtype=torch.bool)
        mask_adjusted[adjusted_token_ids_tensor] = True

        S_before_adjustment = exp_logits[mask_adjusted].sum()
        S_others = exp_logits[~mask_adjusted].sum()

        logits[adjusted_token_ids_tensor] += adjustment_factors
        exp_logits_adjusted = torch.exp(logits[adjusted_token_ids_tensor])
        S_adjusted = exp_logits_adjusted.sum()

        C = torch.log1p((S_before_adjustment - S_adjusted) / S_others) if S_others > 0 else 0.0
        logits[~mask_adjusted] += C

        return logits

class ExpectedValueSearchTree(BaseTokenIdsPrefixTree):
    """
    Implements a search strategy that maintains an upper bound on the expected value
    of paths through each node. Each explored path contributes its continuation probability
    to the upper bound.
    """

    def _additional_node_attributes(self, token_id: Optional[int], node_log_prob: float) -> dict:
        return {
            "total_following_paths_probability": 0.0
        }
    
    def _update_node_metrics(self, node: TokenIdNode, continuation_probability: float, path_length: int, hashed_function_outputs: Optional[int]) -> None:
        """Update node's expected value upper bound by adding the continuation probability."""
        node["total_following_paths_probability"] += continuation_probability

    def _get_adjustment_factor(self, node: TokenIdNode) -> float:
        """
        Get adjustment factor based on expected value upper bound.
        As we explore more failing paths, the upper bound decreases.
        """
        return math.log(1.0 - node["total_following_paths_probability"]) if node["total_following_paths_probability"] < 1.0 else -math.inf

class ExpectedValueSearchTreeWithDiversityPrediction(BaseTokenIdsPrefixTree):
    """
    Implements a search strategy that maintains an upper bound on the expected value
    of paths through each node. Each explored path contributes its continuation probability
    to the upper bound.
    """

    def _additional_node_attributes(self, token_id: Optional[int], node_log_prob: float) -> dict:
        return {
            "total_following_paths_probability": 0.0,
            "function_outputs": set(),
            "num_duplicate_function_outputs" : 0,
            "num_unique_function_outputs": 0
        }
    
    def _update_node_metrics(self, node: TokenIdNode, continuation_probability: float, path_length: int, hashed_function_outputs: Optional[int]) -> None:
        """Update node's expected value upper bound by adding the continuation probability."""
        node["total_following_paths_probability"] += continuation_probability
        if hashed_function_outputs in node["function_outputs"]:
            node["num_duplicate_function_outputs"] += 1
        else:
            node["num_unique_function_outputs"] += 1
            node["function_outputs"].add(hashed_function_outputs)

    def _get_adjustment_factor(self, node: TokenIdNode) -> float:
        """
        Get adjustment factor based on expected value upper bound.
        As we explore more failing paths, the upper bound decreases.
        """
        alpha_prior = 2
        beta_prior = 4

        #Beta distribution
        p_is_duplicate = (alpha_prior + node["num_duplicate_function_outputs"] - 1) / (beta_prior + node["num_duplicate_function_outputs"] + node["num_unique_function_outputs"] - 2)

        return math.log((1.0 - node["total_following_paths_probability"]) * (1 - p_is_duplicate)) if node["total_following_paths_probability"] < 1.0 and p_is_duplicate < 1.0 else -math.inf
    
class BeamSearchLikeTree(BaseTokenIdsPrefixTree):
    """
    Implements a search strategy similar to beam search, maintaining the average
    continuation probability for paths through each node.
    """
    def _additional_node_attributes(self, token_id: Optional[int], node_log_prob: float) -> dict:
        return {
            "avg_continuation_probability": 0.0,
            "path_count": 0
        }
    
    def _update_node_metrics(self, node: TokenIdNode, continuation_probability: float, path_length: int) -> None:
        """
        Update node's average continuation probability using running average formula:
        new_average = old_average + (new_value - old_average) / new_count
        """

        path_normalized_log_probability = math.log(continuation_probability) / path_length
        old_average = node["avg_continuation_probability"]
        node["path_count"] += 1
        node["avg_continuation_probability"] = old_average + (math.exp(path_normalized_log_probability) - old_average) / node["path_count"]

    def _get_adjustment_factor(self, node: TokenIdNode) -> float:
        """
        Get adjustment factor based on average continuation probability.
        Penalizes paths through nodes that typically lead to low probability continuations.
        """
        return math.log(node["avg_continuation_probability"]) if node["avg_continuation_probability"] > 0.0 else 0.0