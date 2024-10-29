import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import networkx as nx




def average_price(game, Agent1, Agent2, a1_list, a2_list):
    """
    Calculate the average price set by each agent over all simulations.

    :return: Tuple of average prices (Agent1, Agent2)
    """
    
    iterations = len(a1_list)

    single_iter_average1 = 0
    single_iter_average2 = 0
    for i in range(iterations):
        # Calculate average price for the stable period in each simulation
        single_iter_average1 += np.sum([a for a in a1_list[i][-int(game.tstable):]])/int(game.tstable)
        single_iter_average2 += np.sum([a for a in a2_list[i][-int(game.tstable):]])/int(game.tstable)
    
    # Calculate overall average prices
    avg_price1 = single_iter_average1 / iterations
    avg_price2 = single_iter_average2 / iterations

    print(f"Average price set by Agent 1 over {iterations} iterations = {avg_price1}")
    print(f"Average price set by Agent 2 over {iterations} iterations = {avg_price2}")

    return avg_price1, avg_price2

def create_heatmap( joint_state_keys, joint_state_counts):
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

def Make_Q( index):
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
            self.single_results.append((all_visited_states, all_actions))

            # Store Q-values if agents use Q-learning or SARSA
            if self.agent1_is_q:
                self.Q_vals_1.append(self.Agent1.Q.copy())
            else:
                self.Q_vals_1.append(None)

            if self.agent2_is_q:
                self.Q_vals_2.append(self.Agent2.Q.copy())
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
    for k in range(len(self.env.init_actions())):
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

def Q_table( index = 0):
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


def create_directed_network_graph( adj_matrix, node_labels):
    if len(node_labels) != len(adj_matrix):
        raise ValueError("The number of labels must match the number of nodes in the matrix.")

    # Create a directed graph from the adjacency matrix
    G = nx.DiGraph(adj_matrix)

    # Relabel nodes with the provided labels
    mapping = {i: node_labels[i] for i in range(len(node_labels))}
    G = nx.relabel_nodes(G, mapping)

    # Set up the plot
    plt.figure(figsize=(10, 10))

    # Use a circular layout for better organization
    pos = nx.spring_layout(G, k=0.5, iterations=20) 
    # pos = nx.kamada_kawai_layout(G, scale = 5)

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=8, font_weight='bold',
            arrows=True, arrowsize=15, edge_color='gray',
            connectionstyle="arc3,rad=0.1")  # Curved edges for clarity

    # Add a title
    plt.title("Directed Network Graph of Agent Actions", fontsize=16)

    # Adjust margins
    plt.tight_layout()

    # Show the plot
    plt.show()

def make_adjency(game, Agent1, Agent2, a1_list, a2_list):



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

def profit_graph(game, Agent1, Agent2, a1_lists, a2_lists):
    # Ensure a1_lists and a2_lists have the same length
    assert len(a1_lists) == len(a2_lists), "a1_lists and a2_lists must have the same length"
    
    num_simulations = len(a1_lists)
    max_length = max(max(len(sim) for sim in a1_lists), max(len(sim) for sim in a2_lists))
    
    # Initialize arrays to store profits
    player1_profits = np.full((num_simulations, max_length), np.nan)
    player2_profits = np.full((num_simulations, max_length), np.nan)
    
    # Calculate profits for each simulation
    for i in range(num_simulations):
        profit_list = [game.compute_profits(np.array([p1,p2])) for p1,p2 in zip(a1_lists[i], a2_lists[i])]
        sim_length = len(profit_list)
        player1_profits[i, :sim_length] = [profit[0] for profit in profit_list]
        player2_profits[i, :sim_length] = [profit[1] for profit in profit_list]
    
    # Calculate average profits, ignoring NaN values
    avg_player1_profits = np.nanmean(player1_profits, axis=0)
    avg_player2_profits = np.nanmean(player2_profits, axis=0)
    
    # Calculate standard deviation for error bars, ignoring NaN values
    std_player1_profits = np.nanstd(player1_profits, axis=0)
    std_player2_profits = np.nanstd(player2_profits, axis=0)
    
    # Create a time list
    time = np.arange(max_length)
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot average profits with error bars
    plt.errorbar(time, avg_player1_profits, yerr=std_player1_profits, label='Player 1', marker='o', capsize=5, capthick=1, elinewidth=1)
    plt.errorbar(time, avg_player2_profits, yerr=std_player2_profits, label='Player 2', marker='s', capsize=5, capthick=1, elinewidth=1)

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Average Profit')
    plt.title('Average Profit over Time for Both Players')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.show()

    # Return the data for further analysis if needed
    return {
        'time': time,
        'avg_player1_profits': avg_player1_profits,
        'avg_player2_profits': avg_player2_profits,
        'std_player1_profits': std_player1_profits,
        'std_player2_profits': std_player2_profits
    }



    






