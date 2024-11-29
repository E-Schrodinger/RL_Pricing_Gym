# main.py

from Environments.IRP import IRP
from Agents.qlearning import Q_Learning
from Agents.dec_qlearning import Dec_Q
from Agents.Exp3 import Exp3
from Agents.SGD import SGD

from Metrics.simulations import Simulations
# from Metrics.Simulation_Base import Simulations
from Metrics.Pricing_Metrics import average_price, profit_graph, simulate_deviation, make_adjacency, state_heatmap, plot_rp, average_price_and_profit

import numpy as np
import os

# Initialize the game environment
game = IRP(tmax=500000, tstable=100000)
print(f'Max val = {game.tmax}')

# Initialize agents
Agent1 = Q_Learning(game, beta=0.0001, Qinit='calvano', cal_k=8, space_type='augment', lump_tol=0.03)
Agent2 = Q_Learning(game, beta=0.0001, Qinit='calvano', cal_k=8)

# Other agent initializations (if needed)
# Agent1_Q = Dec_Q(game, ...)
# Agent2_Q = Dec_Q(game, beta=1, Qinit='calvano', batch_size=100, a1_prices=[1, 1.5, 2, 2.5, 3], space_type='augment', lump_tol=0.05)
# Agent2_exp = Exp3(game, beta=0.0001, Qinit='calvano', cal_k=6, a1_prices=[1, 1.5, 2, 2.5, 3])
# Agent2_PPO = PPO_Agent(...)
Agent2_SGD = SGD(game, action_space_type = "continuous")

# Initialize the Simulations class with the desired number of iterations
SIMULATION_ITERATIONS = 50  # Example: 100 iterations
SM = Simulations(game, Agent2, Agent2_SGD, iterations=SIMULATION_ITERATIONS)

print(Agent1.a1_prices)
print(Agent2_SGD.action_high)
print(Agent2_SGD.action_low)

# Run simulations in parallel
a1_list, a2_list = SM.get_values()

print(f"Final state_space = {Agent1.state_space.shape}")

stats = average_price_and_profit(game, a1_list, a2_list)

# avg_price1 = stats['avg_price1']
# print(f"avg_price1 = {avg_price1}")
# std_price1 = stats['std_price1']
# print(f"std_price1 = {std_price1}")
# avg_price2 = stats['avg_price2']
# print(f"avg_price2 = {avg_price2}")
# std_price2 = stats['std_price2']
# print(f"std_price2 = {std_price2}")
# avg_profit1 = stats['avg_profit1']
# print(f"avg_profit1 = {avg_profit1}")
# std_profit1 = stats['std_profit1']
# print(f"std_profit1 = {std_profit1}")
# avg_profit2 = stats['avg_profit2']
# print(f"avg_profit2 = {avg_profit2}")
# std_profit2 = stats['std_profit2']
# print(f"std_profit2 = {std_profit2}")

# Plot the results
# plot_rp(Agent1_list, Agent2_list, time_step=5000)

# Additional analysis and metrics (optional)
# make_adjacency(...)
# state_heatmap(...)
# simulate_deviation(...)
# find_regret(...)
# average_price(...)
# profit_graph(...)

