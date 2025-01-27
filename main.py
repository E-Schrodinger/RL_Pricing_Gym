# main.py

from Environments.IRP import IRP
from Agents.qlearning import Q_Learning
from Agents.dec_qlearning import Dec_Q
from Agents.Exp3 import Exp3
from Agents.SGD import SGD

from Metrics.simulations import Simulations
# from Metrics.Simulation_Base import Simulations
from Metrics.Pricing_Metrics import average_price, profit_graph, make_adjacency, state_heatmap, plot_rp, average_price_and_profit
from Metrics.Regret_Metric import find_regret

import numpy as np
import os
import time

# Initialize the game environment
game = IRP(tmax=1000000, tstable=100000)
print(f'Max val = {game.tmax}')

# p1,p2 = game.compute_p_competitive_monopoly()
# print(p1)
# print(p2)

d_action_space = [1.0, 1.125,  1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2]

time_step = 1000

# Initialize agents
Agent1 = Q_Learning(game, beta=0.0001, Qinit='uniform', cal_k=8, a1_prices = d_action_space)
Agent2 = Q_Learning(game, beta=0.0001, Qinit='uniform', cal_k=8, a1_prices = d_action_space)

# print(game.compute_profits(np.array([1.473,1.473])))
# print(game.compute_profits(np.array([1.925,1.925])))

# Other agent initializations (if needed)
# Agent1_Q = Dec_Q(game, beta=0.00001, Qinit='uniform', cal_k=8, a1_prices = d_action_space, batch_size = 100)
# Agent2_Q = Dec_Q(game, beta=0.00001, Qinit='uniform', cal_k=8, a1_prices = d_action_space, batch_size = 100)
Agent2_exp = Exp3(game, beta=0.0001, Qinit='uniform', cal_k=8, a1_prices=d_action_space)
# Agent2_PPO = PPO_Agent(...)
# Agent2_SGD = SGD(game, action_space_type = "continuous")

start_time = time.time()

# Initialize the Simulations class with the desired number of iterations
SIMULATION_ITERATIONS = 48  # Example: 100 iterations
SM = Simulations(game, Agent1, Agent2_exp, iterations=SIMULATION_ITERATIONS, ts = time_step, save_agents = False)

print(Agent1.a1_prices)
# print(Agent2_SGD.action_high)
# print(Agent2_SGD.action_low)

# Run simulations in parallel
a1_list, a2_list = SM.get_values()

print(f"Total time = {time.time() - start_time}")

print(f"Final state_space = {Agent1.state_space.shape}")

stats = average_price_and_profit(game, a1_list, a2_list, ts = time_step)

# state_heatmap(game, Agent1, Agent2, a1_list, a2_list, time_step)

# simulate_deviation(game, Agent1, Agent2, 1, 10, deviated_index=0, index=True)

# make_adjacency(Agent1, Agent2, Agent1.Q, Agent2.Q, labels='index', plot_graph=True)

# plot_rp(Agent1_list, Agent2_list, time_step=1)


regret1 = find_regret(game, Agent1, Agent2_exp, a1_list, a2_list, use_loglog=False, regret_type="ratio")
regret2 = find_regret(game, Agent2_exp, Agent1, a2_list, a1_list, use_loglog=False, regret_type="ratio")




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

