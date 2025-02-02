from re import X
from tqdm import tqdm

from shared_utils.code_evaluation.utils import read_samples
from mcts.token_ids_prefix_tree import ExpectedValueSearchTreeWithDiversityPrediction
from os import environ
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl  # Change from pytorch_lightning to lightning.pytorch
from lightning.pytorch.loggers import CSVLogger  # Update import for logger
from value_estimator.train_on_hidden_states_for_next_token import sample_equally_between_solved_and_unsolved
from mcts.value_estimation_network import MultiLayerNN
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from value_estimator.train_on_hidden_states_for_next_token import sample_equally_between_solved_and_unsolved

environ["CUDA_VISIBLE_DEVICES"] = "7"  # todo do this differently

data = read_samples("outputs-samples-meta-llama-Llama-3.1-8B-t0.8.jsonl")
data_2 = read_samples("outputs/samples-t0.8-with-hidden-states.jsonl")
tree_2 = ExpectedValueSearchTreeWithDiversityPrediction.create_from_samples(data_2)
_, y_raw,_ = tree_2.collect_value_estimator_features_and_target(discard_unsolved_problems=False, include_next_token_value_estimates=False)

original_tree = ExpectedValueSearchTreeWithDiversityPrediction.create_from_samples(data)
_, y,_ = original_tree.collect_value_estimator_features_and_target(discard_unsolved_problems=False, include_next_token_value_estimates=False)

import seaborn as sns
import matplotlib.pyplot as plt

_, y_balanced, _ = sample_equally_between_solved_and_unsolved(original_tree, include_next_token_value_estimates=False)
