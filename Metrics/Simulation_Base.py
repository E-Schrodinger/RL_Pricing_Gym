import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import copy  # Import the copy module for deep copying


class Simulations:
    """
    A class to perform and manage multiple simulations of a game environment involving two agents.

    This class handles the initialization of simulations, execution of multiple game iterations,
    and collection of results such as actions taken by agents and their Q-values (if applicable).

    Attributes
    ----------
    iterations : int
        Number of simulation iterations to run (default: 100).
    simulation_results : list or None
        Stores the results of each simulation, including visited states and actions.
    single_results : list or None
        Placeholder for single simulation results (not utilized in current implementation).
    env : object
        The game environment in which the simulations are run.
    Agent1 : object
        The first agent participating in the simulations.
    Agent2 : object
        The second agent participating in the simulations.
    Q_vals_1 : list or None
        Stores Q-values for Agent1 across simulations if Agent1 uses Q-learning or SARSA.
    Q_vals_2 : list or None
        Stores Q-values for Agent2 across simulations if Agent2 uses Q-learning or SARSA.
    Agent1_list : list
        List of Agent1 instances after each simulation.
    Agent2_list : list
        List of Agent2 instances after each simulation.
    agent1_is_q : bool
        Determines if Agent1 uses Q-values (Q-learning or SARSA).
    agent2_is_q : bool
        Determines if Agent2 uses Q-values (Q-learning or SARSA).
    """

    def __init__(self, game, Agent1, Agent2, **kwargs):


        self.iterations = kwargs.get('iterations', 100)
        self.simulation_results = None
        self.single_results = None
        self.env = game
        self.Agent1 = Agent1
        self.Agent2 = Agent2

        self.Q_vals_1 = None
        self.Q_vals_2 = None

        # Initialize lists to store copies of the agents after each simulation
        self.Agent1_list = []
        self.Agent2_list = []

    def has_q_vals(self, agent):
        """
        Check if an agent uses Q-learning or SARSA algorithms.

        Parameters
        ----------
        agent : object
            The agent to check.

        Returns
        -------
        bool
            True if the agent uses Q-learning or SARSA, False otherwise.
        """
        return (type(agent).__name__ in ['Q_Learning', 'Batch_SARSA', 'Dec_Q'] or
                any(base.__name__ in ['Q_Learning', 'Batch_SARSA', 'Dec_Q'] for base in type(agent).__bases__))

    def run_simulations(self):
        """
        Run multiple simulations of the game and store the results.

        This method executes the specified number of simulation iterations, resetting agents
        and running the game environment for each iteration. It collects visited states,
        actions taken, and Q-values if applicable.

        Returns
        -------
        list
            A list containing the results of each simulation, specifically visited states and actions.
        """
        if self.simulation_results is None:
            self.simulation_results = []
            self.Q_vals_1 = []
            self.Q_vals_2 = []

            self.agent1_is_q = self.has_q_vals(self.Agent1)
            self.agent2_is_q = self.has_q_vals(self.Agent2)

            # Run simulations for the specified number of iterations
            for _ in range(int(self.iterations)):
                self.Agent1.reset(self.env)
                self.Agent2.reset(self.env)

                (
                    self.env,
                    s,
                    all_visited_states,
                    all_actions,
                    all_Q1,
                    all_Q2,
                    all_A1,
                    all_A2
                ) = self.env.simulate_game(self.Agent1, self.Agent2, self.env)
                self.simulation_results.append((all_visited_states, all_actions))

                # Store Q-values if agents use Q-learning or SARSA
                if self.agent1_is_q:
                    self.Q_vals_1.append(all_Q1)
                else:
                    self.Q_vals_1.append(None)

                if self.agent2_is_q:
                    self.Q_vals_2.append(all_Q2)
                else:
                    self.Q_vals_2.append(None)

                self.Agent1_list.append(all_A1)
                self.Agent2_list.append(all_A2)
        return self.simulation_results

    def get_values(self):
        """
        Run simulations and retrieve the actions, Q-values, and agent states from all simulations.

        This method ensures that simulations are run and then extracts the actions taken by both
        agents, the Q-values if applicable, and the state of each agent after each simulation.

        Returns
        -------
        tuple
            A tuple containing:
                - a1_list (list of lists): Actions taken by Agent1 across simulations.
                - a2_list (list of lists): Actions taken by Agent2 across simulations.
                - Q1_list (list or None): Q-values for Agent1 across simulations if applicable.
                - Q2_list (list or None): Q-values for Agent2 across simulations if applicable.
                - Agent1_list (list): List of Agent1 instances after each simulation.
                - Agent2_list (list): List of Agent2 instances after each simulation.
        """
        sim_results = self.run_simulations()

        a1_list = []
        a2_list = []
        Q1_list = []
        Q2_list = []
        Agent1_list = []
        Agent2_list = []

        for simulation_idx, (_, all_actions) in enumerate(self.simulation_results):
            # Extract actions for each agent
            a1_actions = [action[0] for action in all_actions]
            a2_actions = [action[1] for action in all_actions]
            a1_list.append(a1_actions)
            a2_list.append(a2_actions)

            Agent1_list.append(self.Agent1_list[simulation_idx])
            Agent2_list.append(self.Agent2_list[simulation_idx])

            # Retrieve Q-values if available
            if self.agent1_is_q:
                Q1_list.append(self.Q_vals_1[simulation_idx])
            else:
                Q1_list.append(None)

            if self.agent2_is_q:
                Q2_list.append(self.Q_vals_2[simulation_idx])
            else:
                Q2_list.append(None)

        # Return the action lists, Q-value lists, and agent lists
        return a1_list, a2_list, Q1_list, Q2_list, Agent1_list, Agent2_list