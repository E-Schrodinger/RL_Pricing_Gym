from Environments.IRP import IRP
from Agents.qlearning import Q_Learning
from Agents.dec_qlearning import Batch_SARSA

import numpy as np

def linear_demand(p):
    d = 2*p+5
    return d

# Init algorithm
game = IRP(tmax = 100000, tstable = 1000, k = 4, beta = 1)
print(f'Max val = {game.tmax}')

Agent1 = Q_Learning(game)
Agent2 = Q_Learning(game)

Agent1_BS = Batch_SARSA(game, batch_size = 10)
Agent2_BS = Batch_SARSA(game, batch_size = 10)

# Compute equilibrium
#_,s = game.simulate_game(Agent1, Agent2, game)
_,s = game.simulate_game(Agent1_BS, Agent2_BS, game)
