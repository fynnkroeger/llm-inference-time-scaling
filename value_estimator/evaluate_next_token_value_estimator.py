import torch
from torch.utils.data import DataLoader
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from statistics import mean, stdev, variance
import random

def calc_stats(data: list[float], metric_name: str) -> dict[str, float]:
    return {
        f"{metric_name}-mean": mean(data),
        f"{metric_name}-variance": variance(data),
        f"{metric_name}-stdev": stdev(data)
    }

def validate_model(model, val_loader: DataLoader, baseline_static_value_predictor: float):
    model.eval()
    token_predictions = []
    comparison_scores = []
    
    with torch.no_grad():
        for hidden_states, token_ids, token_values, expected_values, mask in tqdm(val_loader, desc="Validating"):
            # Get predictions for all vocab
            pred_values = model(hidden_states)  # [batch_size, vocab_size]
            
            # Per-token prediction analysis
            for batch_idx in range(hidden_states.size(0)):
                # Get valid tokens for this example (where mask is 1)
                valid_mask = mask[batch_idx] > 0
                valid_tokens = token_ids[batch_idx][valid_mask]
                true_values = token_values[batch_idx][valid_mask]
                expected_value = expected_values[batch_idx].item()
                
                # Get predictions for the valid tokens
                pred_token_values = pred_values[batch_idx, valid_tokens]
                
                # Store individual token predictions
                for token_id, pred_value, true_value in zip(valid_tokens.tolist(),
                                                          pred_token_values.tolist(),
                                                          true_values.tolist()):
                    token_predictions.append({
                        "token_id": token_id,
                        "predicted_value": pred_value,
                        "true_value": true_value,
                        "error": abs(pred_value - true_value),
                        "baseline_static_predictor_error": abs(pred_value - baseline_static_value_predictor),
                        "expected_value": expected_value
                    })
                
                # Compare token pairs within this example
                if len(valid_tokens) >= 2:
                    for i, j in combinations(range(len(valid_tokens)), 2):
                        true_value_i = true_values[i].item()
                        true_value_j = true_values[j].item()
                        
                        if true_value_i == true_value_j:
                            continue
                            
                        pred_value_i = pred_token_values[i].item()
                        pred_value_j = pred_token_values[j].item()
                        
                        # Scoring based on relative ordering
                        if true_value_i > true_value_j:
                            score = 1 if pred_value_i > pred_value_j else 0
                        else:
                            score = 1 if pred_value_i < pred_value_j else 0
                            
                        comparison_scores.append({
                            "score": score,
                            "random_baseline_score": random.randint(0, 1),
                            "value_difference": abs(true_value_i - true_value_j),
                            "prediction_difference": abs(pred_value_i - pred_value_j)
                        })
    
    # Create summary
    results = {
        "token_predictions": token_predictions,
        "comparison_scores": comparison_scores,
        "metrics": 
            calc_stats([s["score"] for s in comparison_scores], "accuracy") |
            calc_stats([p["error"] for p in token_predictions], "error") |
            calc_stats([s["random_baseline_score"] for s in comparison_scores], "random_baseline_accuary") | {
             "baseline_static_value_predictor": baseline_static_value_predictor,
            "total_comparisons": len(comparison_scores),
            "total_predictions": len(token_predictions)
        }
    }
    
    return results

def evaluate_per_prompt(model, val_dataloader):
    model.eval()
    score_data = []
    scores_value_estimator = []
    
    with torch.no_grad():
        for prompt_hash, prompt_data in tqdm(val_data_by_prompt.items(), 
                                           desc="Evaluating per prompt performance"):
            for (hidden_states, token_ids, token_values, expected_values, metadata) in prompt_data:
                if len(token_ids) < 2:  # Need at least 2 samples to compare
                    continue
                    
                # Get predictions for all tokens
                pred_values = model(hidden_states)  #[vocab_size]
                
                # Compare pairs of predictions
                for i, j in combinations(range(len(token_ids)), 2):
                    # Get relevant tokens and their predicted values
                    tokens_i = token_ids[i]
                    tokens_j = token_ids[j]
                    
                    # Compute average predicted value for each set of tokens
                    pred_value_i = pred_values[tokens_i].item()
                    pred_value_j = pred_values[tokens_j].item()
                    
                    # Get true values
                    true_value_i = token_values[i].item()
                    true_value_j = token_values[j].item()
                    
                    if true_value_i == true_value_j:
                        continue
                    
                    
                    if true_value_i > true_value_j:
                        score = 1 if pred_value_i > pred_value_j else 0
                    else:
                        score = 1 if pred_value_i < pred_value_j else 0
                    
                    scores_value_estimator.append(score)
                    
                    score_data.append({
                        "score": score,
                        "depth": metadata["path_depth"]
                    })
    
    return score_data, scores_value_estimator

# Example usage:
"""
# First run general validation
predictions = predict_value_estimates(model, val_loader)
plot_validation_results(predictions)

# Then run prompt-specific validation
val_data_by_prompt = defaultdict(lambda: ([], [], [], [], []))  # hidden_states, token_ids, token_values, expected_values, metadata

# Populate val_data_by_prompt with your validation data
for hidden_state, token_ids, token_values, expected_value, metadata in validation_data:
    prompt_hash = metadata["prompt_hash"]
    val_data_by_prompt[prompt_hash][0].append(hidden_state)
    val_data_by_prompt[prompt_hash][1].append(token_ids)
    val_data_by_prompt[prompt_hash][2].append(token_values)
    val_data_by_prompt[prompt_hash][3].append(expected_value)
    val_data_by_prompt[prompt_hash][4].append(metadata)

score_data, baseline_scores, model_scores = evaluate_per_prompt(model, val_data_by_prompt)

# Analyze results
print(f"Model accuracy: {mean(model_scores):.3f}")
print(f"Random baseline: {mean(baseline_scores):.3f}")
"""