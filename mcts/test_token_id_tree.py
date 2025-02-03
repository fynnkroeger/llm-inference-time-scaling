from shared_utils.code_evaluation.utils import read_samples
from mcts.token_ids_prefix_tree import ExpectedValueSearchTree
data = read_samples("outputs/samples-meta-llamaLlama-3.2-1B-t0.8.jsonl")
data_single =data[:1]

tree = ExpectedValueSearchTree.create_from_samples(data_single)