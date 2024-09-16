from Environments.IRP import IRP
from Agents.qlearning import Q_Learning
from Agents.dec_qlearning import Batch_SARSA
from Metrics.Pricing_Metrics import Pricing_Metric

import numpy as np


# Init algorithm
game = IRP(tmax = 100000, tstable = 1000, k = 8, beta = 1)
print(f'Max val = {game.tmax}')

Agent1 = Q_Learning(game)
Agent2 = Q_Learning(game)

Agent1_BS = Batch_SARSA(game, batch_size = 10)
Agent2_BS = Batch_SARSA(game, batch_size = 10)

PM = Pricing_Metric(iterations = 2)
#PM.average_price(game, Agent1, Agent2)
PM.state_heatmap(game, Agent1, Agent2)

game.show_stats()

# Compute Q-Values

#game, s = game.simulate_game(Agent1, Agent2, game)
# _,s = game.simulate_game(Agent1, Agent2_BS, game)


# Tests for policy convergence

# print(Agent1.pick_strategies(game, s, 1000)[0])
# print(Agent2.pick_strategies(game, s[::-1], 1000)[0])


