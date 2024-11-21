# main.py

from Environments.IRP import IRP
from Agents.qlearning import Q_Learning
from Agents.dec_qlearning import Dec_Q
from Agents.Exp3 import Exp3
from Agents.SGD import SGD

from Metrics.simulations import Simulations
from Metrics.Pricing_Metrics import average_price, profit_graph, simulate_deviation, make_adjacency, state_heatmap, plot_rp

import numpy as np
import os

# Initialize the game environment
game = IRP(tmax=1000000, tstable=10000)

print(os.cpu_count())
print(f'Max val = {game.tmax}')

# Initialize agents
Agent1 = Q_Learning(game, beta=0.01, Qinit='calvano', cal_k=4, space_type='augment', lump_tol=0.05)
Agent2 = Q_Learning(game, beta=0.01, Qinit='calvano', cal_k=6, a1_prices=[1, 1.5, 2, 2.5, 3])

# Other agent initializations (if needed)
# Agent1_Q = Dec_Q(game, ...)
Agent2_Q = Dec_Q(game, beta=1, Qinit='calvano', batch_size=100, a1_prices=[1, 1.5, 2, 2.5, 3], space_type='augment', lump_tol=0.05)
Agent2_exp = Exp3(game, beta=0.0001, Qinit='calvano', cal_k=6, a1_prices=[1, 1.5, 2, 2.5, 3])
# Agent2_PPO = PPO_Agent(...)
Agent2_SGD = SGD(game, a1_prices=[1, 1.5, 2, 2.5, 3])

# Initialize the Simulations class with the desired number of iterations
SIMULATION_ITERATIONS = 20  # Example: 100 iterations
SM = Simulations(game, Agent1, Agent2, iterations=SIMULATION_ITERATIONS)

# Run simulations in parallel
a1_list, a2_list, Q1_list, Q2_list, Agent1_list, Agent2_list = SM.get_values_parallel()

# Plot the results
plot_rp(Agent1_list, Agent2_list, Q1_list, Q2_list, time_step=5000)

# Additional analysis and metrics (optional)
# make_adjacency(...)
# state_heatmap(...)
# simulate_deviation(...)
# find_regret(...)
# average_price(...)
# profit_graph(...)

