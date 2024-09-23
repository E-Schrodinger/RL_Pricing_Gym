from Environments.IRP import IRP
from Agents.qlearning import Q_Learning
from Agents.dec_qlearning import Dec_Q
from Metrics.Pricing_Metrics import Pricing_Metric
from Metrics.Pricing_Deviations import Pricing_Deviation

import numpy as np


# Init algorithm
game = IRP(tmax = 2000000, tstable = 1000, k = 8)
print(f'Max val = {game.tmax}')
print(game.init_actions())

Agent1 = Q_Learning(game, beta = 0.0001, Qinit = 'calvano')
Agent2 = Q_Learning(game, beta = 0.0001, Qinit = 'calvano')

Agent1_Q = Dec_Q(game, beta = 1, Qinit = 'calvano', batch_size = 100)
Agent2_Q = Dec_Q(game, beta = 1, Qinit = 'calvano', batch_size = 100)

PM = Pricing_Metric(game, Agent1, Agent2, iterations = 5)

### Gives the Average Price
#PM.average_price()

PM.make_adjency()

#print(PM.Q_table(0))

### Makes Heatmap
#PM.state_heatmap()


### Outputs Q-table, 0 = Agent1, 1 = Agent2
#print(PM.Q_table(0))
# print(PM.Q_table(1))

### Outputs the normalized delta value from Calvano: 0 = No collusion, 1 = Full collusion
# print(PM.normalized_delta())

#game.show_stats()


PD = Pricing_Deviation(game, Agent1, Agent2)

### Forces a deviation of agent 1, then outputs a graph which shows the behavior of the agents
### tdeviate = number of iterations for which Agent 1 has a fixed action
### tmax = total number of iterations
### Deviated price = Fixed action. 'Nash' = NE, 'Monopoly' = Monopoly price, you can also input any other integer in the action space

#PD.simulate_deviation(game, tdeviate = 50, tmax = 100, deviated_price = 13)






