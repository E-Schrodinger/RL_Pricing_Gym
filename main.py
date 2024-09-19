from Environments.IRP import IRP
from Agents.qlearning import Q_Learning
from Agents.dec_qlearning import Batch_SARSA
from Metrics.Pricing_Metrics import Pricing_Metric

import numpy as np


# Init algorithm
game = IRP(tmax = 10000, tstable = 1000, k = 4)
print(f'Max val = {game.tmax}')

Agent1 = Q_Learning(game, beta = 1)
Agent2 = Q_Learning(game, beta = 1)

Agent1_BS = Batch_SARSA(game, batch_size = 10)
Agent2_BS = Batch_SARSA(game, batch_size = 10)

PM = Pricing_Metric(game, Agent1, Agent2, iterations = 5)
PM.average_price()
# PM.state_heatmap()
print(PM.Print_Q(0).shape)
print(PM.normalized_delta())

game.show_stats()

# Compute Q-Values

#game, s = game.simulate_game(Agent1, Agent2, game)
# _,s = game.simulate_game(Agent1, Agent2_BS, game)





