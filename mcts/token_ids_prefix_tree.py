from typing import Any, Self, Optional
from abc import ABC, abstractmethod
import math
import warnings
import torch
import numpy as np
from tqdm import tqdm
import random

class TokenIdNode(dict):
    prompt_hash: int
    token_id: Optional[int]  # is None for root
    node_log_prob: float
    total_following_paths_probability: float  # Sum of probabilities from this node to end for all explored paths
    avg_continuation_probability: float  # Average probability from this node to end
    path_count: int    # Number of paths through this node
    path_depth: int # At what height of the tree are we
    children_token_ids: dict[int, Self]
    hidden_states: torch.Tensor
    correct_solutions_counter: int
    false_solutions_counter: int
    parent: Optional[Self]

    def __hash__(self) -> int: # type: ignore
        return hash((self["prompt_hash"], self["token_id"]))
    
    def value_estimator_metadata(self) -> dict[str, Any]:
        return {
            "prompt_hash": self["prompt_hash"],
            "path_depth": self["path_depth"]
        }

    def get_leaf(self) -> Self:
        if len(self["children_token_ids"]) == 0:
            return self
        return self.get_first_child().get_leaf()

    def get_first_child(self) -> Self:
        if len(self["children_token_ids"]) == 0:
            raise Exception(f"Node: {self} has no children!")
        return list(self["children_token_ids"].values())[0]
    
    def value_estimator_features(self) -> Optional[np.ndarray]:
        if self["hidden_states"] is None:
            return None
        # Up cast from half precision to full precision https://stackoverflow.com/questions/78128662/converting-pytorch-bfloat16-tensors-to-numpy-throws-typeerror
        return self["hidden_states"].float().numpy()
    
    def value_estimator_target(self, include_next_token_value_estimates: bool = False):
        if include_next_token_value_estimates:
            return self.value_estimator_target(include_next_token_value_estimates=False), list(self["children_token_ids"].keys()), list(x.value_estimator_target(include_next_token_value_estimates=False) for x in self["children_token_ids"].values())
        else:
            return self["correct_solutions_counter"] / (self["correct_solutions_counter"] + self["false_solutions_counter"])
    
    def collect_value_estimator_features_and_target(self, discard_unsolved_problems: bool = False, include_next_token_value_estimates: bool=False) -> tuple[list[np.ndarray], list, list[dict]]:
        X,y, metadata = [], [], []
        for child_node in self["children_token_ids"].values():
            
            X_,y_, metadata_ = child_node.collect_value_estimator_features_and_target(discard_unsolved_problems=discard_unsolved_problems, include_next_token_value_estimates=include_next_token_value_estimates)
            
            X += X_
            y += y_
            metadata += metadata_
        
        # The last node won't have a hidden state itself
        if  self["correct_solutions_counter"] +  self["false_solutions_counter"] > 0:
            if self.value_estimator_features() is not None and (not discard_unsolved_problems or self["correct_solutions_counter"] > 0):
                X.append(self.value_estimator_features())
                y.append(self.value_estimator_target(include_next_token_value_estimates=include_next_token_value_estimates))
                metadata.append(self.value_estimator_metadata())
        else:
            pass
            #warn(f"Node has no solution counter: {self}" )
        return X,y, metadata
    
    def _update_total_following_paths_probability(self) -> None:
        if len(self["children_token_ids"]) == 0:
            self["total_following_paths_probability"] = 1.0
        else:
            total_following_paths_probability = 0.0
            for node in self["children_token_ids"].values():
                total_following_paths_probability += math.exp(node["node_log_prob"]) * node["total_following_paths_probability"]
            self["total_following_paths_probability"] = total_following_paths_probability

        if self["total_following_paths_probability"] > 1.0:
            warnings.warn(f"{self} total_following_paths_probability > 1.0. This shouldnt happen because then it was probably decoded twice: {self['total_following_paths_probability']} ")

    def __repr__(self) -> str:
        return f"TokenIdNode(token_id:{self["token_id"]}, log_prob:{self["node_log_prob"]}, children_token_ids:{self["children_token_ids"].keys()}, correct_solutions_counter:{self["correct_solutions_counter"]}, false_solutions_counter:{self["false_solutions_counter"]})"

class BaseTokenIdsPrefixTree(ABC):
    @classmethod
    def create_from_samples(cls, samples: list[dict]) -> Self:
        tree = cls()
        for sample in samples:
            raw_logprobs = []
            output_token_ids = []
            for x in sample["logprobs"]:
                raw_logprobs.append(x["logprob"])
                output_token_ids.append(x["token_id"])
            tree.add_sequence(sample["prompt_token_ids"], output_token_ids, raw_logprobs, sample["hidden_states"],hash(tuple(sample["function_outputs"])), sample["passed"])
       
        return tree

    def number_of_unique_prompts(self) -> int:
        return len(self.prompt_root)
    
    def split(self, n_prompts_in_first_tree: int) -> tuple[Self, Self]:
        assert n_prompts_in_first_tree < self.number_of_unique_prompts()
        warnings.warn("This methode does not copy/deepclone anything and just 'moves pointers'. You should throw away any references to the old tree because they still reference the same objects!")
        prompts = list(self.prompt_root.keys())
        random.shuffle(prompts)

        tree_1 = self.__class__()
        tree_2 = self.__class__()

        for key in prompts[:n_prompts_in_first_tree]:
            tree_1.prompt_root[key] = self.prompt_root[key]
        
        for key in prompts[n_prompts_in_first_tree:]:
            tree_2.prompt_root[key] = self.prompt_root[key]
        
        return tree_1, tree_2


    def collect_value_estimator_features_and_target(self, discard_unsolved_problems: bool = True, include_next_token_value_estimates: bool=False) -> tuple[list[np.ndarray], list, list[dict]]:
        X,y, metadata  = [], [], []
        for node in tqdm(self.prompt_root.values(), "Loading features and target"):
            X_root,y_root, metadata_ = node.collect_value_estimator_features_and_target(discard_unsolved_problems=discard_unsolved_problems,include_next_token_value_estimates=include_next_token_value_estimates)
            X += X_root
            y += y_root
            metadata += metadata_
        return X,y, metadata
    
    def __init__(self) -> None:
        self.prompt_root: dict[tuple[int,...], TokenIdNode] = {}
        self.iteration = 0
        self.metrics = {
            "common_prefix_ratio_sum": 0.0,
            "total_duplicates": 0,
            "total_sequences": 0,
            "potentially_saved_prefix_tokens": 0
        }

    def calculate_metrics(self) -> dict:
        return {
            "common_prefix_ratio": self.metrics["common_prefix_ratio_sum"] / self.metrics["total_sequences"],
            "totaL_duplicates": self.metrics["total_duplicates"],
            "total_seqeunces": self.metrics["total_sequences"],
            "p_is_duplicate": self.metrics["total_duplicates"] / self.metrics["total_sequences"],
            "potentially_saved_prefix_tokens": self.metrics["potentially_saved_prefix_tokens"]
        }

    def _create_empty_node(self, token_id: Optional[int], node_log_prob: float, prompt_token_ids_hash: int, hidden_states: Optional[torch.Tensor], is_correct: Optional[bool], height: int, parent: Optional[TokenIdNode]) -> TokenIdNode:
        node =  TokenIdNode({
            "prompt_hash": prompt_token_ids_hash,
            "token_id": token_id,
            "node_log_prob": node_log_prob,
            "children_token_ids": {},
            "hidden_states": hidden_states,
            "correct_solutions_counter": 1 if is_correct and is_correct is not None else 0,
            "false_solutions_counter": 1 if not is_correct and is_correct is not None else 0,
            "path_depth": height,
            "parent": parent,
            "total_following_paths_probability": 1.0
        })

        node.update(self._additional_node_attributes(token_id, node_log_prob))
        return node
    
    
    def _additional_node_attributes(self, token_id: Optional[int], node_log_prob: float) -> dict:
        return {

        }

    @abstractmethod
    def _update_node_metrics(self, node: TokenIdNode, path_length: int, hashed_function_outputs:Optional[int]) -> None:
        """Update the node's metrics based on the specific search strategy."""
        pass

    @abstractmethod
    def _get_adjustment_factor(self, node: TokenIdNode) -> float:
        """Calculate the adjustment factor for logits based on the specific search strategy."""
        pass

    def add_sequence(self, prompt_token_ids: list[int], token_ids: list[int], log_probs: list[float], hidden_states: Optional[torch.Tensor] = None, hashed_function_outputs : Optional[int]=None, is_correct: Optional[bool] = None) -> None:
        assert len(token_ids) == len(log_probs), "Each token_id must have one log_prob. However, the two lists have different lengths"

        prompt_token_ids_as_tuple = tuple(prompt_token_ids)
        if prompt_token_ids_as_tuple not in self.prompt_root:
            self.prompt_root[prompt_token_ids_as_tuple] = self._create_empty_node(None, 0.0,hash(prompt_token_ids_as_tuple), None, None, 0, None)

        is_duplicate = True
        number_of_duplicate_tokens = 0

        node = self.prompt_root[prompt_token_ids_as_tuple]
        for i in range(len(token_ids)):
            self._update_node_metrics(node, len(token_ids) - i, hashed_function_outputs)
            if token_ids[i] not in node["children_token_ids"]:
                is_duplicate = False
                node["children_token_ids"][token_ids[i]] = self._create_empty_node(token_ids[i], log_probs[i],hash(prompt_token_ids_as_tuple),  None, None, i + 1, node)
            else:
                number_of_duplicate_tokens += 1
                
            if hidden_states is not None:
                node["hidden_states"] = hidden_states[i]
            if is_correct is not None:
                if is_correct:
                    node["correct_solutions_counter"] += 1
                else:
                    node["false_solutions_counter"] += 1
            node = node["children_token_ids"][token_ids[i]]
            node["node_log_prob"] = log_probs[i]
            
        # set correctness counter for last node
        self._update_node_metrics(node, len(token_ids), hashed_function_outputs)
        if is_correct is not None:
            if is_correct:
                node["correct_solutions_counter"] += 1
            else:
                node["false_solutions_counter"] += 1

        # Backtrack and update total_following_paths_probability
        for i in range(len(token_ids) -1, -1, -1):
            node._update_total_following_paths_probability()
            node = node["parent"]

        self.metrics["common_prefix_ratio_sum"] += number_of_duplicate_tokens / len(token_ids)
        self.metrics["potentially_saved_prefix_tokens"] += number_of_duplicate_tokens
        if is_duplicate:
            self.metrics["total_duplicates"] += 1
        self.metrics["total_sequences"] += 1

    """
    Call this method to pass a signal to the tree that we finished adding new sequences for the current generation iteration.
    Subclasses can use this method inorder to start things like training value estimation models
    """
    def finished_adding_sequences_watermark(self) -> None:
        self.iteration += 1

    def adjust_logits_fast(self, prompt_token_ids: list[int], output_token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        prompt_token_ids_as_tuple = tuple(prompt_token_ids)

        if prompt_token_ids_as_tuple not in self.prompt_root:
            return logits

        node: TokenIdNode = self.prompt_root[prompt_token_ids_as_tuple]
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
    
    def _update_node_metrics(self, node: TokenIdNode, path_length: int, hashed_function_outputs: Optional[int]) -> None:
        pass

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
    
    def _update_node_metrics(self, node: TokenIdNode, path_length: int, hashed_function_outputs: Optional[int]) -> None:
        """Update node's expected value upper bound by adding the continuation probability."""
        if hashed_function_outputs in node["function_outputs"]:
            node["num_duplicate_function_outputs"] += 1
        else:
            node["num_unique_function_outputs"] += 1
            node["function_outputs"].add(hashed_function_outputs)

    #list(tree.prompt_root.values())[0].get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child()..get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child()..get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child().get_first_child()
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