import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import networkx as nx

class Pricing_Metric:
    """
    A class to analyze pricing strategies in a multi-agent game environment.
    """

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
            for _ in range(int(self.iterations)):
                self.Agent1.reset(self.env)
                self.Agent2.reset(self.env)
                self.env, s, all_visited_states, all_actions = self.env.simulate_game(self.Agent1, self.Agent2, self.env)
                self.simulation_results.append((all_visited_states, all_actions))

                # Store Q-values if agents use Q-learning or SARSA
                if self.agent1_is_q:
                    self.Q_vals_1.append(self.Agent1.Q[0].copy())
                else:
                    self.Q_vals_1.append(None)

                if self.agent2_is_q:
                    self.Q_vals_2.append(self.Agent2.Q[0].copy())
                else:
                    self.Q_vals_2.append(None)

    def make_action_lists(self):
        self.run_simulations()

        a1_list = []
        a2_list = []

        for _, all_actions in self.simulation_results:
            a1_actions = [action[0] for action in all_actions]
            a2_actions = [action[1] for action in all_actions]
            a1_list.append(a1_actions)
            a2_list.append(a2_actions)

        return a1_list, a2_list
    
    def average_price(self):
        """
        Calculate the average price set by each agent over all simulations.

        :return: Tuple of average prices (Agent1, Agent2)
        """
        self.run_simulations()

        single_iter_average1 = 0
        single_iter_average2 = 0
        for all_visited_states, all_actions in self.simulation_results:
            # Calculate average price for the stable period in each simulation
            single_iter_average1 += np.sum([self.env.init_actions()[a[0]] for a in all_actions[-int(self.env.tstable):]])/int(self.env.tstable)
            single_iter_average2 += np.sum([self.env.init_actions()[a[1]] for a in all_actions[-int(self.env.tstable):]])/int(self.env.tstable)
        
        # Calculate overall average prices
        avg_price1 = single_iter_average1 / self.iterations
        avg_price2 = single_iter_average2 / self.iterations

        print(f"Average price set by Agent 1 over {self.iterations} iterations = {avg_price1}")
        print(f"Average price set by Agent 2 over {self.iterations} iterations = {avg_price2}")

        return avg_price1, avg_price2

    def create_heatmap(self, joint_state_keys, joint_state_counts):
        """
        Create a heatmap of joint state distributions.

        :param joint_state_keys: List of joint state tuples
        :param joint_state_counts: List of counts for each joint state
        """
        # Find the maximum state indices
        max_state1 = max(key[0] for key in joint_state_keys) + 1
        max_state2 = max(key[1] for key in joint_state_keys) + 1

        # Create a 2D array for the heatmap
        heatmap_data = np.zeros((max_state2, max_state1))

        # Fill the heatmap data
        for key, count in zip(joint_state_keys, joint_state_counts):
            heatmap_data[key[1], key[0]] = count

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap_data, cmap='YlOrRd')

        # Set title and labels
        ax.set_title('Joint State Distribution Heatmap')
        ax.set_xlabel('Agent 1 States')
        ax.set_ylabel('Agent 2 States')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('State Count', rotation=-90, va="bottom")

        # Add text annotations
        for i in range(max_state2):
            for j in range(max_state1):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.0f}',
                            ha="center", va="center", color="black")

        # Set ticks
        ax.set_xticks(np.arange(max_state1))
        ax.set_yticks(np.arange(max_state2))
        ax.set_xticklabels(np.arange(max_state1))
        ax.set_yticklabels(np.arange(max_state2))

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

    def state_heatmap(self):
        """
        Generate a heatmap of joint state distributions.

        :return: List of joint state counts
        """
        self.run_simulations()

        sdim, _ = self.env.init_state()

        # Initialize joint state counts
        joint_state_counts = [0] * (sdim[0] * sdim[0])
        joint_state_keys = [(i, j) for i in range(sdim[0]) for j in range(sdim[0])]

        # Count joint states in the stable period of each simulation
        for all_visited_states, _ in self.simulation_results:
            joint_counted = all_visited_states[-int(self.env.tstable):]
            joint_count = Counter(joint_counted)
            
            for i, key in enumerate(joint_state_keys):
                joint_state_counts[i] += joint_count.get(key, 0)

        self.create_heatmap(joint_state_keys, joint_state_counts)

        return joint_state_counts
    
    def Make_Q(self, index):
        """
        Create a DataFrame of Q-values for the specified agent.

        :param index: 0 for Agent1, 1 for Agent2
        :return: DataFrame of Q-values
        """
        if self.simulation_results == None and self.single_results == None:
            self.single_results = []
            self.Q_vals_1 = []
            self.Q_vals_2 = []

            self.agent1_is_q = self.has_q_vals(self.Agent1)
            self.agent2_is_q = self.has_q_vals(self.Agent2)

            # Run simulations for the specified number of iterations
            for _ in range(1):
                self.Agent1.reset(self.env)
                self.Agent2.reset(self.env)
                self.env, s, all_visited_states, all_actions = self.env.simulate_game(self.Agent1, self.Agent2, self.env)
                self.simulation_results.append((all_visited_states, all_actions))

                # Store Q-values if agents use Q-learning or SARSA
                if self.agent1_is_q:
                    self.Q_vals_1.append(self.Agent1.Q[0].copy())
                else:
                    self.Q_vals_1.append(None)

                if self.agent2_is_q:
                    self.Q_vals_2.append(self.Agent2.Q[0].copy())
                else:
                    self.Q_vals_2.append(None)

        else:
            self.run_simulations()

        if index == 0:
            Qvals = self.Q_vals_1[-1]
        elif index == 1:
            Qvals = self.Q_vals_2[-1]

        sdim, _ = self.env.init_state()

        column_names = []

        # Generate state indices
        state_indices = []
        for i in range(sdim[0]):
                for j in range(sdim[0]):
                    state_indices.append((i,j))
        agent1_states = [state[0] for state in state_indices]
        agent2_states = [state[1] for state in state_indices]

        array_list = [agent1_states,agent2_states]

        # Create column names
        for i in range(self.env.n):
            column_names.append(f"Agent {i+1} State")

        if index == 1:
            column_names.reverse()

        # Add Q-values for each action
        for k in range(Qvals.shape[2]):
            column_names.append(f"{k}")

            action_list = []

            for i in range(sdim[0]):
                for j in range(sdim[0]):
                    action_list.append(Qvals[(i,j)][k])
            
            array_list.append(action_list)
        
        # Create DataFrame
        data_dict = {name: arr for name, arr in zip(column_names, array_list)}
        df = pd.DataFrame(data_dict)

        return df

    def Q_table(self, index = 0):
        """
        Return the Q-values for the specified agent.

        :param index: 0 for Agent1, 1 for Agent2
        :return: DataFrame of Q-values or None if agent doesn't use Q-learning
        """
        if index == 0:
            Agent = self.Agent1
        elif index == 1:
            Agent = self.Agent2
        
        if self.has_q_vals(Agent):
            generated_Q_vals = self.Make_Q(index)
            return generated_Q_vals
        else:
            print(f"Agent {index + 1} has no Q Values")
            return None
        
    def normalized_delta(self):
        """
        Calculate the normalized delta between average price and competitive/monopoly prices.

        :return: Normalized delta value
        """
        p_competitive, p_monopoly = self.env.compute_p_competitive_monopoly()
        avg_price1, avg_price2 = self.average_price()
        pi_hat = (avg_price1+avg_price2)/self.env.n

        return (pi_hat - p_competitive)/(p_monopoly-p_competitive)
    
    def create_directed_network_graph(self, adj_matrix, node_labels):
        if len(node_labels) != len(adj_matrix):
            raise ValueError("The number of labels must match the number of nodes in the matrix.")

        # Create a directed graph from the adjacency matrix
        G = nx.DiGraph(adj_matrix)

        # Relabel nodes with the provided labels
        mapping = {i: node_labels[i] for i in range(len(node_labels))}
        G = nx.relabel_nodes(G, mapping)

        # Set up the plot
        plt.figure(figsize=(20, 20))

        # Use a circular layout for better organization
        pos = nx.spring_layout(G, k=0.3, iterations=50) 

        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=8, font_weight='bold',
                arrows=True, arrowsize=15, edge_color='gray',
                connectionstyle="arc3,rad=0.1")  # Curved edges for clarity

        # Add a title
        plt.title("Directed Network Graph of Agent Actions", fontsize=16)

        # Adjust margins
        plt.tight_layout()

        # Show the plot
        plt.show()

    def make_adjency(self):
        if self.simulation_results == None and self.single_results == None:
            self.single_results = []
            self.Q_vals_1 = []
            self.Q_vals_2 = []

            self.agent1_is_q = self.has_q_vals(self.Agent1)
            self.agent2_is_q = self.has_q_vals(self.Agent2)

            # Run simulations for the specified number of iterations
            for _ in range(1):
                self.Agent1.reset(self.env)
                self.Agent2.reset(self.env)
                self.env, s, all_visited_states, all_actions = self.env.simulate_game(self.Agent1, self.Agent2, self.env)
                self.single_results.append((all_visited_states, all_actions))

                # Store Q-values if agents use Q-learning or SARSA
                if self.agent1_is_q:
                    self.Q_vals_1.append(self.Agent1.Q[0].copy())
                else:
                    self.Q_vals_1.append(None)

                if self.agent2_is_q:
                    self.Q_vals_2.append(self.Agent2.Q[0].copy())
                else:
                    self.Q_vals_2.append(None)

        else:
            self.run_simulations()

    
        Qvals1 = self.Q_vals_1[-1]
    
        Qvals2 = self.Q_vals_2[-1]


        adj_matrix = np.zeros((self.env.k*self.env.k, self.env.k*self.env.k))

        node_names = []

        for i in range(self.env.k):
            for j in range(self.env.k):
                node_names.append(f"({i},{j})")

        for i in range(self.env.k):
            for j in range(self.env.k):
            #     print(Qvals1[tuple((i,j))])
            #     print(Qvals2[tuple((i,j))])
                x = (i*(self.env.k) + j)
                y = (np.argmax(Qvals1[tuple((i,j))])*(self.env.k) + np.argmax(Qvals2[tuple((i,j))]))
                # print(f"x = {x}, y = {y}")
                adj_matrix[x,y] = 1
        
        self.create_directed_network_graph(adj_matrix, node_names)

        






