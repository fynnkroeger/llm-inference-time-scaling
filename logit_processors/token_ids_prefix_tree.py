from typing import Self, TypedDict, Union, Optional
import math
import torch

class TokenIdNode(TypedDict):
    token_id: Optional[int] # is None for root
    node_log_prob: float
    all_following_paths_prob_sum: float 
    children_token_ids: dict[int, Self]

class TokenIdsPrefixTree:
    def __init__(self) -> None:
        self.prompt_root: dict[tuple[int,...], TokenIdNode] = {}

    
    def add_sequence(self, prompt_token_ids: list[int], token_ids: list[int], log_probs: list[float]) -> None:
        assert len(token_ids) == len(log_probs), "Each token_id must have one log_prob. However, the two lists have different lengths"
        partial_sequeunce_log_prob_sums = [0.0] * len(log_probs)
        log_prob_sum = 0
        for i in range(len(log_probs) - 1, -1, -1):
            log_prob_sum += log_probs[i]
            partial_sequeunce_log_prob_sums[i] = log_prob_sum
        

        prompt_token_ids_as_tuple = tuple(prompt_token_ids)
        if prompt_token_ids_as_tuple not in self.prompt_root:
            self.prompt_root[prompt_token_ids_as_tuple] = {"token_id": None, "node_log_prob": 0.0, "all_following_paths_prob_sum": 0.0, "children_token_ids": {}}

        node = self.prompt_root[prompt_token_ids_as_tuple]
        for i in range(len(token_ids)):
            node["all_following_paths_prob_sum"] += math.exp(partial_sequeunce_log_prob_sums[i])
            if token_ids[i] not in node["children_token_ids"]:
                node["children_token_ids"][token_ids[i]] = {"token_id": token_ids[i], "node_log_prob": log_probs[i], "all_following_paths_prob_sum": 0.0, "children_token_ids": {}}
            node = node["children_token_ids"][token_ids[i]]


    def adjust_logits_fast(self, prompt_token_ids: list[int], output_token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        """
        Fast chatgpt implementation. Adjusts logits for the provided sequence of token IDs based on stored logprob adjustments.

        Args:
            prompt_token_ids (list[int]): Sequence of token IDs corresponding to the logits.
            output_token_ids (torch.Tensor): Tensor of token IDs for the generated output.
            logits (torch.Tensor): Tensor of logits to be adjusted.

        Returns:
            torch.Tensor: The adjusted logits.
        """
        prompt_token_ids_as_tuple = tuple(prompt_token_ids)

        if prompt_token_ids_as_tuple not in self.prompt_root:
            return logits

        node = self.prompt_root[prompt_token_ids_as_tuple]
        for token_id in output_token_ids:
            if token_id not in node["children_token_ids"]:
                return logits
            node = node["children_token_ids"][token_id]

        # Get adjusted token IDs and their corresponding adjustment factors
        adjusted_token_ids = list(node["children_token_ids"].keys())
        adjustment_factors = torch.tensor(
            [math.log(1.0 - x["all_following_paths_prob_sum"]) if x["all_following_paths_prob_sum"] < 1.0 else -math.inf for x in node["children_token_ids"].values()],
            device=logits.device
        )

        # Convert to tensor for GPU operations
        adjusted_token_ids_tensor = torch.tensor(adjusted_token_ids, device=logits.device, dtype=torch.long)

        # Calculate S_before_adjustment and S_others
        exp_logits = torch.exp(logits)
        mask_adjusted = torch.zeros_like(logits, dtype=torch.bool)
        mask_adjusted[adjusted_token_ids_tensor] = True

        S_before_adjustment = exp_logits[mask_adjusted].sum()
        S_others = exp_logits[~mask_adjusted].sum()

        # Adjust logits for specified token IDs
        logits[adjusted_token_ids_tensor] += adjustment_factors
        exp_logits_adjusted = torch.exp(logits[adjusted_token_ids_tensor])
        S_adjusted = exp_logits_adjusted.sum()

        # Compute normalization constant
        C = torch.log1p((S_before_adjustment - S_adjusted) / S_others) if S_others > 0 else 0.0

        # Apply normalization to other logits
        logits[~mask_adjusted] += C

        return logits
    
    def adjust_logits(self, prompt_token_ids: list[int], output_token_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # My reference implementation
        prompt_token_ids_as_tuple = tuple(prompt_token_ids)

        if prompt_token_ids_as_tuple not in self.prompt_root:
            return logits
        
        node = self.prompt_root[prompt_token_ids_as_tuple]
        for token_id in output_token_ids:
            if token_id not in node["children_token_ids"]:
                return logits
            node = node["children_token_ids"][token_id]
        
        
        adjusted_token_ids = node["children_token_ids"].keys()
        adjusted_token_ids_set= set(adjusted_token_ids)
        adjustment_factors = [math.log(1.0 - x["all_following_paths_prob_sum"]) for x in node["children_token_ids"].values()]

        

        # Calculate S_others (sum of exponentials of other logits)
        exp_logits = torch.exp(logits)
        
        
        S_before_adjustment = 0
        S_adjusted = 0
        S_others = 0

        for i, x in enumerate(exp_logits):
            if i in adjusted_token_ids_set:
                S_before_adjustment += x
            else:
                S_others += x

        
        for token_id, adjustment in zip(adjusted_token_ids, adjustment_factors):
            logits[token_id] += adjustment
            S_adjusted += math.exp(logits[token_id])

        # Compute normalization constant
        C = math.log(1 + ( (S_before_adjustment - S_adjusted) / S_others)) if S_others > 0 else 0

        # Adjust logits
        for token_id in range(len(logits)):
            if token_id not in adjusted_token_ids_set:
                logits[token_id] += C
        
        return logits