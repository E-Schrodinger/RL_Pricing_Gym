from Environments.IRP import IRP
from Agents.dec_qlearning import simulate_game


def linear_demand(p):
    d = 2*p+5
    return d

# Init algorithm
game = IRP(tmax = 10000, tstable = 100, k = 4)
print(f'Max val = {game.tmax}')

# Compute equilibrium
game_equilibrium = simulate_game(game, batch_size= 2)