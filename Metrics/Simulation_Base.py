import sys
import numpy as np
import copy  # Import the copy module for deep copying


def check_end(game, t, stable1, stable2):
    """
    Check if the game has converged.

    Parameters
    ----------
    game : IRP
        The game environment.
    t : int
        Current iteration number.
    stable1 : int
        Number of stable periods for algorithm 1.
    stable2 : int
        Number of stable periods for algorithm 2.

    Returns
    -------
    bool
        True if the game has converged, False otherwise.
    """
    if (t % game.tstable == 0) & (t > 0):
        sys.stdout.write("\rt=%i " % t)
        sys.stdout.flush()
    if stable1 > game.tstable and stable2 > game.tstable:
        print('Both Algorithms Converged!')
        return True
    if t == game.tmax - 1:
        if stable1 > game.tstable:
            print("Algorithm 1 : Converged. Algorithm 2: Not Converged")
            return True
        elif stable2 > game.tstable:
            print("Algorithm 1 : Not Converged. Algorithm 2: Converged")
            return True

        print('ERROR! Not Converged!')
        return True
    return False


def simulate_game(Agent1, Agent2, game, ts=1):
    """
    Simulate the game between two agents.

    Parameters
    ----------
    Agent1 : object
        First agent with pick_strategies and update_function methods.
    Agent2 : object
        Second agent with pick_strategies and update_function methods.
    game : IRP
        The game environment.
    ts : int
        Sampling rate: store results every ts timesteps.

    Returns
    -------
    tuple
        A tuple containing:
        - game: The game environment after simulation.
        - s: Final state.
        - all_visited_states: List of visited states at sampled timesteps.
        - all_actions: List of actions taken at sampled timesteps.
        - all_A1: List of copies of Agent1 at sampled timesteps.
        - all_A2: List of copies of Agent2 at sampled timesteps.
    """
    s = (Agent1.s0, Agent2.s0)
    stable1 = 0
    stable2 = 0
    stable_state0 = 0
    stable_state1 = 0
    all_visited_states = []
    all_actions = []
    all_A1 = []
    all_A2 = []

    for t in range(int(game.tmax)):
        a1 = Agent1.pick_strategies(game, s, t)
        a2 = Agent2.pick_strategies(game, s[::-1], t)
        a = (a1, a2)
        a_prof = np.array([a1, a2])

        pi1 = game.compute_profits(a_prof)
        s1 = a

        same_state0 = (s[0] == s1[0])
        stable_state0 = (stable_state0 + same_state0) * same_state0

        same_state1 = (s[1] == s1[1])
        stable_state1 = (stable_state1 + same_state1) * same_state1

        _, stable1 = Agent1.update_function(game, s, a, pi1[0], stable1, t)
        _, stable2 = Agent2.update_function(game, s[::-1], a[::-1], pi1[1], stable2, t)
        s = s1

        if t % ts == 0:
            all_actions.append(a)
            all_visited_states.append(s1)
            all_A1.append(copy.deepcopy(Agent1))
            all_A2.append(copy.deepcopy(Agent2))
            

        if check_end(game, t, stable1, stable2):
            Agent1.final_price = a[0]
            Agent2.final_price = a[1]
            break

    

    return game, s, all_visited_states, all_actions, all_A1, all_A2


class Simulations:
    """
    A class to perform and manage multiple simulations of a game environment involving two agents.

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
    Agent1_list : list
        List of Agent1 instances after each simulation.
    Agent2_list : list
        List of Agent2 instances after each simulation.
    save_agents : bool
        Whether to save the agents' states during simulations.
    ts : int
        Sampling rate for storing actions and agents.
    """

    def __init__(self, game, Agent1, Agent2, **kwargs):
        self.iterations = kwargs.get('iterations', 100)
        self.ts = kwargs.get('ts', 1)  # Sampling rate
        self.save_agents = kwargs.get('save_agents', False)
        self.simulation_results = None
        self.single_results = None
        self.env = game
        self.Agent1 = Agent1
        self.Agent2 = Agent2

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

        Returns
        -------
        list
            A list containing the results of each simulation, specifically visited states and actions.
        """
        if self.simulation_results is None:
            self.simulation_results = []

            # Run simulations for the specified number of iterations
            for _ in range(int(self.iterations)):
                self.Agent1.reset(self.env)
                self.Agent2.reset(self.env)

                (
                    self.env,
                    s,
                    all_visited_states,
                    all_actions,
                    all_A1,
                    all_A2
                ) = simulate_game(self.Agent1, self.Agent2, self.env, ts=self.ts)
                self.simulation_results.append((all_visited_states, all_actions))

                if self.save_agents:
                    self.Agent1_list.append(all_A1)
                    self.Agent2_list.append(all_A2)
        return self.simulation_results

    def get_values(self):
        """
        Run simulations and retrieve the actions and agent states from all simulations.

        Returns
        -------
        tuple
            A tuple containing:
                - a1_list (list of lists): Actions taken by Agent1 across simulations.
                - a2_list (list of lists): Actions taken by Agent2 across simulations.
                - Agent1_list (list): List of Agent1 instances at sampled timesteps (if save_agents is True).
                - Agent2_list (list): List of Agent2 instances at sampled timesteps (if save_agents is True).
        """
        sim_results = self.run_simulations()

        a1_list = []
        a2_list = []

        if self.save_agents:
            Agent1_list = []
            Agent2_list = []

        for simulation_idx, (all_visited_states, all_actions) in enumerate(self.simulation_results):
            # Extract actions for each agent
            a1_actions = [action[0] for action in all_actions]
            a2_actions = [action[1] for action in all_actions]
            a1_list.append(a1_actions)
            a2_list.append(a2_actions)

            # Collect agent instances after simulation
            if self.save_agents:
                Agent1_list.append(self.Agent1_list[simulation_idx])
                Agent2_list.append(self.Agent2_list[simulation_idx])

        # Return the collected values
        if self.save_agents:
            return a1_list, a2_list, Agent1_list, Agent2_list
        else:
            return a1_list, a2_list
