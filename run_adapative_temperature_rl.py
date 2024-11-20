import numpy as np
import random
import math
import os
from collections import OrderedDict
os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_T = 100
MIN_T = 0

MIN_SOLVED = 0
MAX_SOLVED = 164
s = []

STEP = 8

for t in range(MIN_T, MAX_T + 1, STEP):
    s += [(i,t) for i in range(MIN_SOLVED, MAX_SOLVED + 1, STEP)]

#s = [(i, t) for i,t  in zip(range(MIN_SOLVED, MAX_SOLVED + 1), range(MIN_T, MAX_T + 1))]#[2**i for i in range(0, 7)]

a = [i/10 for i in range(0, 26)]


s_0 = s[0]


sigma = np.std(s)
epsilon = 0.1
learning_rate = 0.5



a_min = min(a)
a_max = max(a)

def normalize(v, min, max):
    return (v - min) / (max - min)

from sklearn.preprocessing import PolynomialFeatures
def basis_function(s_i: tuple[int, int], a_i: float):
    s_normalized= normalize(s_i[0], MIN_SOLVED, MAX_SOLVED)
    a_normalized = normalize(a_i, a_min, a_max)
    t_normalized = normalize(s_i[1], MIN_T, MAX_T)

    poly = PolynomialFeatures(degree=5)
        
    return poly.fit_transform(np.array([[s_normalized, a_normalized, t_normalized]]))[0]

M =  len(basis_function((1,1), 0.0))

weights = np.array(object=[random.random() for _ in range(M)])

def parametic_q(s: tuple[int, int], a: float, weights):
    return np.dot(np.array(basis_function(s, a)), weights)

def epsilon_greedy_action(s: tuple[int, int]) -> float:
    if random.random() <= epsilon:
        i = random.randrange(0, len(a))
        return a[i]
    else:
        max_reward = -math.inf
        chosen_a = a[0]
        for a_i in a:
            reward = parametic_q(s, a_i, weights)
            if reward > max_reward:
                max_reward = reward
                chosen_a = a_i
        return chosen_a


from rl.adaptive_temperature.simulation import simulate, simulate_with_probabilities
from rl.adaptive_temperature.visualize_policy import visualize_2d_state_space_policy, visualize_policy

from sklearn.linear_model import LinearRegression, Ridge

K = 80
q_target_values = OrderedDict()

from collections import Counter
q_value_update_counter = Counter()

MAX_Q_VALUES = 20 * 10
for i in range(K):
    new_q_target_values = simulate(s[0], s, epsilon_greedy_action, parametic_q, weights)
    q_value_update_counter.update(Counter(new_q_target_values.keys()))
    for s_a, r in new_q_target_values.items():
        if s_a in q_target_values:
            q_target_values[s_a] = (1 - learning_rate) * q_target_values[s_a] + learning_rate * r
        else:
            q_target_values[s_a] = r
    
    
    print(epsilon, learning_rate)
    X = np.array([basis_function(x[0], x[1]) for x in q_target_values.keys()])
    y = np.array(list(q_target_values.values()))

    reg = LinearRegression(fit_intercept=False).fit(X, y)
    weights = reg.coef_
    visualize_2d_state_space_policy(s, a, parametic_q, weights, q_target_values, f"outputs/rl/adaptive_temperature/policy-wow.png")
    visualize_2d_state_space_policy(s, a, parametic_q, weights, q_target_values, f"outputs/rl/adaptive_temperature/policy-wow-{i}.png")
    epsilon -= epsilon/K
    learning_rate -= learning_rate/K

    while len(q_target_values) > MAX_Q_VALUES:
        q_target_values.popitem(last=False)