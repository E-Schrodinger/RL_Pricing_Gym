import numpy as np
import copy  # For deep copying agents and environment
from joblib import Parallel, delayed
import os
from .Simulation_Base import simulate_game

class Simulations:
    """
    A class to perform and manage multiple simulations of a game environment involving two agents.

    Attributes
    ----------
    iterations : int
        Number of simulation iterations to run (default: 100).
    save_agents : bool
        Whether to save the agents' states during simulations.
    ts : int
        Sampling rate for storing actions and agents.
    simulation_results : list
        Stores the results of each simulation, including visited states and actions.
    env : object
        The game environment in which the simulations are run.
    Agent1 : object
        The first agent participating in the simulations.
    Agent2 : object
        The second agent participating in the simulations.
    Agent1_list : list
        List of Agent1 instances at sampled timesteps.
    Agent2_list : list
        List of Agent2 instances at sampled timesteps.
    agent1_is_q : bool
        Determines if Agent1 uses Q-values (Q-learning or SARSA).
    agent2_is_q : bool
        Determines if Agent2 uses Q-values (Q-learning or SARSA).
    """

    def __init__(self, game, Agent1, Agent2, **kwargs):
        self.iterations = kwargs.get('iterations', 100)
        self.save_agents = kwargs.get('save_agents', False)
        self.ts = kwargs.get('ts', 1)  # Sampling rate
        self.simulation_results = []
        self.env = game
        self.Agent1 = Agent1
        self.Agent2 = Agent2
        self.Agent1_list = []
        self.Agent2_list = []
        self.agent1_is_q = self.has_q_vals(self.Agent1)
        self.agent2_is_q = self.has_q_vals(self.Agent2)

    def has_q_vals(self, agent):
        """
        Check if an agent uses Q-learning or SARSA algorithms.
        """
        return (type(agent).__name__ in ['Q_Learning', 'Batch_SARSA', 'Dec_Q'] or
                any(base.__name__ in ['Q_Learning', 'Batch_SARSA', 'Dec_Q'] for base in type(agent).__bases__))

    def single_simulation(self, sim_index):
        """
        Perform a single simulation iteration.

        Parameters
        ----------
        sim_index : int
            The index of the simulation iteration (can be used for seeding reproducibility).

        Returns
        -------
        dict
            A dictionary containing simulation results.
        """
        # Deep copy the environment and agents to ensure thread safety
        env_copy = copy.deepcopy(self.env)
        agent1_copy = copy.deepcopy(self.Agent1)
        agent2_copy = copy.deepcopy(self.Agent2)

        # Reset the agents with the copied environment
        agent1_copy.reset(env_copy)
        agent2_copy.reset(env_copy)

        # Run the simulation
        (
            env_result,
            s,
            all_visited_states,
            all_actions,
            all_A1,
            all_A2
        ) = simulate_game(agent1_copy, agent2_copy, env_copy, ts=self.ts)

        # Prepare the result
        result = {
            'visited_states': all_visited_states,
            'actions': all_actions,
            'Agent1': all_A1 if self.save_agents else None,
            'Agent2': all_A2 if self.save_agents else None
        }

        return result

    def run_simulations_parallel(self):
        """
        Run multiple simulations in parallel and store the results.

        Returns
        -------
        None
        """
        # Determine the number of jobs (processes) to run in parallel
        num_jobs = min(self.iterations, os.cpu_count())

        # Use joblib's Parallel and delayed to execute simulations in parallel
        parallel = Parallel(n_jobs=num_jobs, prefer="threads")  # prefer="processes" can be used alternatively

        # Create a list of delayed simulation calls
        simulation_results = parallel(delayed(self.single_simulation)(i) for i in range(self.iterations))

        # Aggregate the results
        for res in simulation_results:
            self.simulation_results.append((res['visited_states'], res['actions']))

            if self.save_agents:
                self.Agent1_list.append(res['Agent1'])
                self.Agent2_list.append(res['Agent2'])

    def get_values(self):
        """
        Run simulations in parallel and retrieve the actions and agent states from all simulations.

        Returns
        -------
        tuple
            A tuple containing:
                - a1_list (list of lists): Actions taken by Agent1 across simulations.
                - a2_list (list of lists): Actions taken by Agent2 across simulations.
                - Agent1_list (list): List of Agent1 instances at sampled timesteps (if save_agents is True).
                - Agent2_list (list): List of Agent2 instances at sampled timesteps (if save_agents is True).
        """
        # Run the simulations in parallel
        self.run_simulations_parallel()

        # Initialize lists to collect values
        a1_list = []
        a2_list = []

        if self.save_agents:
            Agent1_list = []
            Agent2_list = []

        # Iterate through the simulation results and extract required information
        for simulation_idx, (all_visited_states, all_actions) in enumerate(self.simulation_results):
            # Extract actions for each agent
            a1_actions = [action[0] for action in all_actions]
            a2_actions = [action[1] for action in all_actions]
            a1_list.append(a1_actions)
            a2_list.append(a2_actions)

            if self.save_agents:
                Agent1_list.append(self.Agent1_list[simulation_idx])
                Agent2_list.append(self.Agent2_list[simulation_idx])

        # Return the collected values
        if self.save_agents:
            return a1_list, a2_list, Agent1_list, Agent2_list
        else:
            return a1_list, a2_list