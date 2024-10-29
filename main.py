from Environments.IRP import IRP
from Agents.qlearning import Q_Learning
from Agents.dec_qlearning import Dec_Q

from Metrics.Simulation_Base import Simulations
from Metrics.Pricing_Metrics import average_price, profit_graph
from Metrics.Pricing_Deviations import Pricing_Deviation
from Metrics.Regret_Metric import find_regret

import numpy as np


# Init algorithm
game = IRP(tmax = 2000000, tstable = 10000)
print(f'Max val = {game.tmax}')


Agent1 = Q_Learning(game, beta = 0.001, Qinit = 'calvano', cal_k = 4, space_type = 'augment', lump_tol = 0.05)
Agent2 = Q_Learning(game, beta = 0.001, Qinit = 'calvano', cal_k = 6, a1_prices = [1,1.5,2,2.5,3])


# Agent1_Q = Dec_Q(game, beta = 1, Qinit = 'calvano', batch_size = 100)
Agent2_Q = Dec_Q(game, beta = 1, Qinit = 'calvano', batch_size = 100, a1_prices = [1,1.5,2,2.5,3], space_type = 'augment', lump_tol = 0.05)



print(Agent1.a1_space)
print(Agent2.a1_space)
SM = Simulations(game, Agent1, Agent2_Q, iterations = 2)

### Gives the Average Price
# PM.average_price()

a1_list, a2_list, Q1_list, Q2_list = SM.get_values()
print(Agent1.price_state_space)
print(Agent2.price_state_space)




find_regret(game, Agent1, Agent2, a1_list, a2_list, use_loglog= False, regret_type='ratio')

# average_price(game, Agent1, Agent2, a1_list, a2_list)
# profit_graph(game, Agent1, Agent2, a1_list, a2_list)





