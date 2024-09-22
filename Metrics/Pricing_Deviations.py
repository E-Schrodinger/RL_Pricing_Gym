import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from Environments.IRP import IRP

class Pricing_Deviation:
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

    def run_simulations(self):
        """
        Run multiple simulations of the game and store the results.
        """
        self.simulation_results = []
        self.Q_vals_1 = []
        self.Q_vals_2 = []

        self.Agent1.reset(self.env)
        self.Agent2.reset(self.env)
        self.env, s, all_visited_states, all_actions = self.env.simulate_game(self.Agent1, self.Agent2, self.env)
        self.simulation_results.append((all_visited_states, all_actions))
        return s

    def simulate_deviation(self, game, tdeviate, tmax, deviated_price = 'Nash'):
        s = self.run_simulations()
        all_visited_states = []
        all_actions = []
        a1_values = []
        a2_values = []
        stable1 = 0
        stable2 = 0
        p_competitive, p_monopoly = self.env.compute_p_competitive_monopoly()

        if deviated_price == "Nash":
            deviation = 0
        elif deviated_price == "Monopoly":
            deviation = game.K-1
        else:
            deviation = deviated_price

        for t in range(int(tmax)):
           # print(f"t = {t} ----------------------------------------------------------------------------------------")
            
            if t<= tdeviate:
                a1 = deviation
                a2 = self.Agent2.pick_strategies(game, s[::-1], t)[0]
            else:
                a1 = self.Agent1.pick_strategies(game, s, t)[0]
                a2 = self.Agent2.pick_strategies(game, s[::-1], t)[0]
            a = (a1, a2)
            all_actions.append(a)
            a1_values.append(a1)
            a2_values.append(a2)
            # print(a)
            pi1 = game.PI[a]
            pi2 = game.PI[a[::-1]]
            s1 = a

            _, stable1 = self.Agent1.update_function(game, s, a, s1, pi1, stable1, t)
            _, stable2 = self.Agent2.update_function(game, s[::-1], a[::-1], s1[::-1], pi2, stable2, t)
            s = s1
            all_visited_states.append(s1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(a1_values)), a1_values, label='Agent 1')
        plt.plot(range(len(a2_values)), a2_values, label='Agent 2')
        plt.xlabel('Time')
        plt.ylabel('Action Value')
        plt.title('Agent Actions Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
 
        return game, s, all_visited_states, all_actions

