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

environ["CUDA_VISIBLE_DEVICES"] = "7"  # todo do this differently

data = read_samples("outputs-samples-meta-llama-Llama-3.1-8B-t0.8.jsonl")

original_tree = ExpectedValueSearchTreeWithDiversityPrediction.create_from_samples(data)
tree_train, tree_val = original_tree.split(int(2/3 * 164))


results = []
for n_samples in [10_000, 20_000, 40_000, 80_000, 160_000, 320_000]:
    X_train, y_train, _ = sample_equally_between_solved_and_unsolved(tree_train, n_data_points=n_samples, include_next_token_value_estimates=False)
    X_val_raw, y_val_raw, metadata_val = sample_equally_between_solved_and_unsolved(tree_val, include_next_token_value_estimates=False)

    print(f"Train size: {len(X_train)} Val size: {len(X_val_raw)}")
    
    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    X_val, y_val = torch.Tensor(X_val_raw), torch.Tensor(y_val_raw)

    # Split into training and validation datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    input_size = 4096
    # DataLoaders for batching
    batch_size = 512
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    # Initialize the model
    model = MultiLayerNN(input_size, [256, 128, 64, 32])

    early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    # Train the model
    trainer = pl.Trainer(max_epochs=25, logger=CSVLogger("outputs/"), callbacks=[early_stop, checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    best_model = MultiLayerNN.load_from_checkpoint(checkpoint_callback.best_model_path)
    print("Best model device:",best_model.device)
    best_model.to("cuda:0")
    # Split validation by the prompt hash (should be unique the same way task id is. But we dont fit to our datastructure of a dataset with task ids)
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
    

    score_data = []
    prediction_data = []
    best_model.eval()
    from itertools import combinations
    with torch.no_grad():
        for (X_val, y_targets, m_i) in tqdm(val_data_per_task, "Evaluting per task id decider performance"):
            X_val = X_val.to("cuda:0")
            preds = best_model(X_val).squeeze().tolist()
            if len(y_targets) < 2:
                continue
            for i in range(len(y_targets)):
                random_baseline_prediction = random.random()
                prediction_data.append({
                    "error": abs(y_targets[i] - preds[i]),
                    "value": y_targets[i],
                    "prediction":  preds[i],
                    "random_baselin_error": abs(y_targets[i] - random_baseline_prediction),
                    "path_depth": m_i[i]["path_depth"]
                })
            # Gives only the unique pairs: list(combinations(range(2), 2)) == [(0, 1)]
            for i, j in combinations(range(len(y_targets)), 2):
                if i == j or y_targets[i] == y_targets[j]:
                    continue
                
                path_depth_difference = abs(m_i[i]["path_depth"] - m_i[j]["path_depth"])
                if path_depth_difference > 5:
                    continue
                mean_path_depth = (m_i[i]["path_depth"] + m_i[j]["path_depth"]) / 2 
                x_1 = preds[i]
                x_2 = preds[j]
                
                score = None
                scores_random_baseline = None
                if y_targets[i] > y_targets[j]:
                    scores_random_baseline = 1
                    if x_1 > x_2:
                        score = 1
                    else:
                        score = 0
                else:
                    scores_random_baseline = 0
                    if x_1 < x_2:
                        score = 1
                    else:
                        score = 0

                score_data.append({
                    "score": score,
                    "random_baseline_score": scores_random_baseline,
                    "mean_path_depth": mean_path_depth,
                    "depth_1": m_i[i]["path_depth"],
                    "depth_2": m_i[j]["path_depth"],
                    "path_depth_difference": path_depth_difference
                })

    df_score = pd.DataFrame.from_records(score_data)
    df_error  = pd.DataFrame.from_records(prediction_data)
    df_score.to_csv(f"outputs/single_value_estimator/scores_df-mlp-big-{len(X_train)}.csv")
    df_error.to_csv(f"outputs/single_value_estimator/error-df-mlp-big-{len(X_train)}.csv")

    print("mean prediction error:", df_error["error"].mean())
    print("mean score:", df_score["score"].mean())