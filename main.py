from Environments.IRP import IRP
from Agents.qlearning import simulate_game


def linear_demand(p):
    d = 2*p+5
    return d

# Init algorithm
game = IRP(tmax = 100, tstable = 10)

# Compute equilibrium
game_equilibrium = simulate_game(game, batch_size= 2)