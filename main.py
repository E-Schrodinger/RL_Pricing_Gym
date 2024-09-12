from Environments.IRP import IRP
from Agents.qlearning import Q_Learning


def linear_demand(p):
    d = 2*p+5
    return d

# Init algorithm
game = IRP(tmax = 100000, tstable = 1000)
print(f'Max val = {game.tmax}')

Agent1 = Q_Learning(game)
Agent2 = Q_Learning(game)

# Compute equilibrium
game.simulate_game(Agent1, Agent2, game)