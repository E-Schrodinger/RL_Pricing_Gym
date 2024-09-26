import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from Environments.IRP import IRP

class Regret_Metric:
    def __init__(self, game, Agent1, Agent2, **kwargs):
        """
        Initialize the Pricing_Metric object.

        :param game: The game environment
        :param Agent1: The first agent
        :param Agent2: The second agent
        :param kwargs: Additional parameters (e.g., number of iterations)
        """
        self.simulation_results = None
        self.env = game
        self.Agent1 = Agent1
        self.Agent2 = Agent2


    def find_regret(self, a1_list, a2_list):
    

        action_space = len(self.env.init_actions())
        # print(f"Action space: {action_space}")
        
        # Calculate total rewards for each action
        total_rewards = np.array([
            sum(self.env.PI[tuple([i , a2_list[j]])][0]
                for j in range(len(a2_list)))
            for i in range(action_space)
        ])
        print(total_rewards)

        best_strategy = np.argmax(total_rewards)

        # Calculate cumulative sums
        best_rewards = np.cumsum([
            self.env.PI[tuple([best_strategy , 
                            a2_list[j]])][0]
            for j in range(len(a2_list))
        ])
        
        actual_rewards = np.cumsum([
            self.env.PI[tuple([a1_list[j] , a2_list[j]])][0]
            for j in range(len(a2_list))
        ])
        
        # Calculate regret
        regret_over_time = 1 - (actual_rewards / best_rewards)
        
        return regret_over_time.tolist()

    def plot_regret(self, regret_over_time):
            plt.figure(figsize=(10, 6))
            plt.plot(regret_over_time)
            plt.xlabel('Time steps')
            plt.ylabel('Regret')
            plt.title('Regret over Time')
            plt.grid(True)
            plt.show()

    # def find_regret(self, a1_list, a2_list, index = 0):
    #     if index == 0:
    #         list1 = a1_list
    #         list2 = a2_list
    #     elif index == 1:
    #         list1 = a2_list
    #         list2 = a1_list

    #     action_space = len(self.env.init_actions())
    #     single_strategy = []

    #     for i in range(action_space):
    #         total_reward = 0
    #         for j in range(len(list1)):
    #             temp_pair = [0,0]
    #             temp_pair[index] = i
    #             temp_pair[1-index] = list2[j]
    #             total_reward += self.env.PI[tuple(temp_pair)]
    #         single_strategy.append(total_reward)
        
    #     best_strategy = np.argmax(single_strategy)

    #     regret_over_time = []

    #     for i in range(len(list1)):
    #         total_reward_i = 0
    #         best_reward_i = 0
    #         for j in range(i+1):
    #             temp_pair = [0,0]
    #             temp_pair[index] = best_strategy
    #             temp_pair[1-index] = list2[j]
    #             best_reward_i += self.env.PI[tuple(temp_pair)]
    #             total_reward_i += self.env.PI[(list1,list2)]
    #         regret_over_time.append(1-(total_reward_i/best_reward_i))


    #     return regret_over_time
    
   
    

