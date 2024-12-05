# from Environments.IRP import IRP
# from Agents.qlearning import Q_Learning
# from Agents.dec_qlearning import Dec_Q

# from Metrics.Simulation_Base import Simulations
# from Metrics.Pricing_Metrics import Pricing_Metric
# from Metrics.Pricing_Deviations import Pricing_Deviation
# from Metrics.Regret_Metric import Regret_Metric

import numpy as np

from Agents.QBase import QBase

# game = IRP(tmax = 2000000, tstable = 10000, k = 4)
# Agent1 = Q_Learning(game, beta = 0.1, Qinit = 'calvano', k =4)
# Agent2 = Q_Learning(game, beta = 0.1, Qinit = 'calvano', k = 4)



# QB = QBase(game=game, cal_k=4)
# print(QB.Q.shape)
# # print(QB.Q_vals[(0,1)])

# new_slice = np.zeros((4,1,4))

# result_array = np.hstack((QB.Q, new_slice))
# result_array = np.hstack((result_array, new_slice))

# print(result_array[3,5])

a = np.array([1,2,3])
b = np.array([9,9,9])

print(np.argmax(a))
