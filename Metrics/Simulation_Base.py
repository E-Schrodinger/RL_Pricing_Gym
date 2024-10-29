import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange



class Simulations:
    def __init__(self, game, Agent1, Agent2, **kwargs):
        """
        Initialize the Pricing_Metric object.

        :param game: The game environment
        :param Agent1: The first agent
        :param Agent2: The second agent
        :param kwargs: Additional parameters (e.g., number of iterations)
        """
        self.iterations = kwargs.get('iterations', 100)
        self.simulation_results = None
        self.single_results = None
        self.env = game
        self.Agent1 = Agent1
        self.Agent2 = Agent2

        self.Q_vals_1 = None
        self.Q_vals_2 = None

    def has_q_vals(self, agent):
        """
        Check if an agent uses Q-learning or SARSA.

        :param agent: The agent to check
        :return: Boolean indicating if the agent uses Q-values
        """
        return (type(agent).__name__ in ['Q_Learning', 'Batch_SARSA', 'Dec_Q'] or
                any(base.__name__ in ['Q_Learning', 'Batch_SARSA', 'Dec_Q'] for base in type(agent).__bases__))

    
    # @njit(parallel=True)
    def run_simulations(self):
        """
        Run multiple simulations of the game and store the results.
        """
        if self.simulation_results is None:
            self.simulation_results = []
            self.Q_vals_1 = []
            self.Q_vals_2 = []

            self.agent1_is_q = self.has_q_vals(self.Agent1)
            self.agent2_is_q = self.has_q_vals(self.Agent2)

            # Run simulations for the specified number of iterations
            for _ in prange(int(self.iterations)):
                self.Agent1.reset(self.env)
                self.Agent2.reset(self.env)
                self.env, s, all_visited_states, all_actions = self.env.simulate_game(self.Agent1, self.Agent2, self.env)
                self.simulation_results.append((all_visited_states, all_actions))

                # Store Q-values if agents use Q-learning or SARSA
                if self.agent1_is_q:
                    self.Q_vals_1.append(self.Agent1.Q.copy())
                else:
                    self.Q_vals_1.append(None)

                if self.agent2_is_q:
                    self.Q_vals_2.append(self.Agent2.Q.copy())
                else:
                    self.Q_vals_2.append(None)

    def get_values(self):
        """
        Run simulations and return lists of actions and Q-values for both agents.

        :return: Tuple containing:
                - a1_list: List of actions taken by Agent1 across simulations
                - a2_list: List of actions taken by Agent2 across simulations
                - Q1_list: List of Q-values for Agent1 (None if Agent1 doesn't use Q-values)
                - Q2_list: List of Q-values for Agent2 (None if Agent2 doesn't use Q-values)
        """
        self.run_simulations()

        a1_list = []
        a2_list = []
        Q1_list = []
        Q2_list = []

        for simulation_idx, (_, all_actions) in enumerate(self.simulation_results):
            # Extract actions for each agent
            a1_actions = [action[0] for action in all_actions]
            a2_actions = [action[1] for action in all_actions]
            a1_list.append(a1_actions)
            a2_list.append(a2_actions)

            # Retrieve Q-values if available
            if self.agent1_is_q:
                Q1_list.append(self.Q_vals_1[simulation_idx])
            else:
                Q1_list.append(None)

            if self.agent2_is_q:
                Q2_list.append(self.Q_vals_2[simulation_idx])
            else:
                Q2_list.append(None)

        return a1_list, a2_list, Q1_list, Q2_list