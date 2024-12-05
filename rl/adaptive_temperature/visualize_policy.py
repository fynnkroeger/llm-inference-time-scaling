import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def visualize_policy(
    state_space: list[int],
    action_space: list,
    parametric_q,
    weights,
    q_target_values: dict,
    file_name_policy: Optional[str] = None,
    file_name_q_function: Optional[str] = None
):
    # Determine optimal policy based on q_target_values
    optimal_actions = []
    real_policy = []
    q_values_grid = np.zeros((len(action_space), len(state_space)))  # Grid to store Q-values for heatmap
    

    for i, s in enumerate(state_space):
        # Compute Q-values using the parametric function for the real policy
        q_values_real = [parametric_q(s, a, weights) for a in action_space]
        real_policy_action = action_space[np.argmax(q_values_real)]
        real_policy.append(real_policy_action)

        # Compute Q-values from q_target_values for the optimal policy
        q_values_observed = [
            q_target_values.get((s, a), float('-inf')) for a in action_space
        ]
        optimal_action = action_space[np.argmax(q_values_observed)] if any(v != float('-inf') for v in q_values_observed) else None
        optimal_actions.append(optimal_action if optimal_action is not None else real_policy_action)
        
        # Store Q-values for heatmap
        for j, q_value in enumerate(q_values_real):
            q_values_grid[j, i] = q_value

    # Define color scale based on observed Q-values
    sampled_q_values = list(q_target_values.values())
    min_q, max_q = min(sampled_q_values), max(sampled_q_values)

    # Visualize both policies on the same plot
    plt.figure()
    plt.plot(state_space, optimal_actions, marker='o', label="Optimal Policy")
    plt.plot(state_space, real_policy,
             marker='x', linestyle='--', color='red', label="Real Policy (Parametric)")
    plt.xlabel("State")
    plt.ylabel("Action")
    plt.title("Optimal Policy vs. Real Policy Visualization")
    plt.legend()
    plt.savefig(file_name_policy or "policy.png")

    # Visualize Q-function heatmap with sampled Q-values overlaid
    plt.figure()
    plt.imshow(q_values_grid, cmap='viridis', aspect='auto', origin='lower', vmin=min_q, vmax=max_q)
    plt.colorbar(label="Q-value")
    plt.xlabel("State")
    plt.ylabel("Action")
    plt.title("Q-Function Heatmap")
    plt.xticks(ticks=range(len(state_space)), labels=state_space)
    plt.yticks(ticks=range(len(action_space)), labels=[f"{a:.2f}" for a in action_space])

    # Overlay sampled Q-values
    sampled_states = [s for (s, _) in q_target_values.keys()]
    sampled_actions = [a for (_, a) in q_target_values.keys()]
    plt.scatter(
        [state_space.index(s) for s in sampled_states],
        [action_space.index(a) for a in sampled_actions],
        c=sampled_q_values,
        cmap='viridis',
        edgecolor='black',
        marker='o',
        s=100,
        vmin=min_q,
        vmax=max_q,
        label="Sampled Q-values"
    )
    plt.legend(loc="upper right")
    
    # Save and show the Q-function heatmap
    plt.savefig(file_name_q_function or "q_function_heatmap.png")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional

def visualize_2d_state_space_policy(
    state_space: list[tuple[int, int]],
    action_space: list,
    parametric_q,
    weights,
    q_target_values: dict,
    file_name_policy: Optional[str] = None,
    file_name_q_function: Optional[str] = None
):
    # Determine optimal actions based on q_target_values
    optimal_actions = []
    
    for s in state_space:
        # Compute Q-values using the parametric function for each action
        q_values_real = [parametric_q(s, a, weights) for a in action_space]
        optimal_action = action_space[np.argmax(q_values_real)]
        optimal_actions.append(optimal_action)

    # Extract state dimensions for plotting
    state_x = [s[0] for s in state_space]
    state_y = [s[1] for s in state_space]
    
    # Map each optimal action to an index for color-coding
    action_indices = [action_space.index(action)/10 for action in optimal_actions]
    
    # Create a DataFrame for plotting with Seaborn
    data = pd.DataFrame({
        'State X': state_x,
        'State Y': state_y,
        'Optimal Action': action_indices
    })
    
    # Create the Seaborn scatter plot
    plt.figure(figsize=(10, 8))
    scatter_plot = sns.scatterplot(
        data=data,
        x='State X',
        y='State Y',
        hue='Optimal Action',
        palette='viridis',
        edgecolor='black',
        s=100,
        marker='o'
    )
    scatter_plot.legend(title="Action Index")
    plt.xlabel("Number of Solved Items")
    plt.ylabel("Timestamps")
    plt.title("2D State Space Optimal Policy Visualization (Color by Action)")
    
    # Save and show the plot
    if file_name_policy:
        plt.savefig(file_name_policy)
    plt.show()

