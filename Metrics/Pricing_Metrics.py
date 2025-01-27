import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import networkx as nx
from scipy import stats
from Environments.IRP import IRP
import seaborn as sns


def average_price(game, a1_list, a2_list, ts):
    """
    Calculate the average price set by each agent over all simulations.

    This function computes the average price that each agent sets during the 
    stable period of each simulation. The stable period is defined by the last
    `tstable` time steps of each simulation.

    Parameters
    ----------
    game : object
        The game environment containing simulation parameters such as `tstable`.
    Agent1 : object
        The first agent participating in the simulations.
    Agent2 : object
        The second agent participating in the simulations.
    a1_list : list of lists
        A list where each sublist contains the actions taken by Agent1 in a simulation.
    a2_list : list of lists
        A list where each sublist contains the actions taken by Agent2 in a simulation.

    Returns
    -------
    tuple
        A tuple containing the average prices set by Agent1 and Agent2 respectively.
        Format: (avg_price1, avg_price2)
    """
    
    iterations = len(a1_list)

    single_iter_average1 = 0
    single_iter_average2 = 0
    for i in range(iterations):
        # Calculate average price for the stable period in each simulation
        single_iter_average1 += np.sum([a for a in a1_list[i][-int(game.tstable/ts):]]) / int(game.tstable/ts)
        single_iter_average2 += np.sum([a for a in a2_list[i][-int(game.tstable/ts):]]) / int(game.tstable/ts)
    
    # Calculate overall average prices
    avg_price1 = single_iter_average1 / iterations
    avg_price2 = single_iter_average2 / iterations

    print(f"Average price set by Agent 1 over {iterations} trajecotories = {avg_price1}")
    print(f"Average price set by Agent 2 over {iterations} trajecotories = {avg_price2}")

    return avg_price1, avg_price2




def average_price_and_profit(game, a1_list, a2_list, ts):
    """
    Calculate the average and standard deviation of prices set by each agent,
    as well as the average profit and standard deviation of profits for each agent
    over all simulations during the stable period. Additionally, compute the
    average and standard deviation of Delta values for each agent, where Delta is defined as:
    
        Δ_i ≡ ( \bar{π}_i - π_i^N ) / ( π_i^M - π_i^N )
    
    for Agent i (i = 1, 2),
    
    where:
    - \(\bar{\pi}_i\) is the average profit of Agent i upon convergence.
    - \(\pi_i^N\) is the profit of Agent i in the Bertrand-Nash static equilibrium.
    - \(\pi_i^M\) is the profit of Agent i under full collusion (monopoly).
    
    Parameters
    ----------
    game : object
        The game environment containing simulation parameters and necessary methods such as `compute_p_competitive_monopol` and `compute_profits`.
    a1_list : list of lists
        A list where each sublist contains the actions (prices) taken by Agent1 in a simulation.
    a2_list : list of lists
        A list where each sublist contains the actions (prices) taken by Agent2 in a simulation.
    ts : float
        Time step or scaling factor used to compute the number of stable actions.
    
    Returns
    -------
    dict
        A dictionary containing:
            - 'avg_price1': Average price set by Agent1.
            - 'std_price1': Standard deviation of prices set by Agent1.
            - 'avg_price2': Average price set by Agent2.
            - 'std_price2': Standard deviation of prices set by Agent2.
            - 'avg_profit1': Average profit for Agent1.
            - 'std_profit1': Standard deviation of profits for Agent1.
            - 'avg_profit2': Average profit for Agent2.
            - 'std_profit2': Standard deviation of profits for Agent2.
            - 'avg_delta1': Average Delta value for Agent1.
            - 'std_delta1': Standard deviation of Delta values for Agent1.
            - 'avg_delta2': Average Delta value for Agent2.
            - 'std_delta2': Standard deviation of Delta values for Agent2.
    """
    
    iterations = len(a1_list)

    if iterations == 0:
        raise ValueError("The action lists are empty. Please provide valid simulation data.")

    # Compute Nash equilibrium prices and monopoly prices
    p_competitive, p_monopoly = game.compute_p_competitive_monopoly()

    # Compute profits for Nash equilibrium and monopoly
    pi_N = game.compute_profits(p_competitive)  # [pi1_N, pi2_N]
    pi_M = game.compute_profits(p_monopoly)     # [pi1_M, pi2_M]
    
    # Extract individual Nash and Monopoly profits
    pi1_N, pi2_N = pi_N
    pi1_M, pi2_M = pi_M

    # Check to prevent division by zero in Delta calculation for each agent
    if np.isclose(pi1_M, pi1_N):
        raise ValueError("Monopoly profit and Nash equilibrium profit for Agent 1 are too close, causing division by zero in Delta calculation.")
    if np.isclose(pi2_M, pi2_N):
        raise ValueError("Monopoly profit and Nash equilibrium profit for Agent 2 are too close, causing division by zero in Delta calculation.")

    # Lists to store per-simulation statistics
    list_avg_p1 = []
    list_avg_p2 = []
    list_std_p1 = []
    list_std_p2 = []
    list_pi1 = []
    list_pi2 = []
    list_delta1 = []
    list_delta2 = []
    
    for i in range(iterations):
        # Extract the last tstable actions for each agent
        num_stable_actions = max(int(game.tstable / ts), 1)  # Ensure at least one action is taken
        stable_actions_a1 = a1_list[i][-num_stable_actions:]
        stable_actions_a2 = a2_list[i][-num_stable_actions:]
        
        if len(stable_actions_a1) == 0 or len(stable_actions_a2) == 0:
            raise ValueError(f"Simulation {i} does not have enough stable actions. Ensure tstable and ts are set correctly.")
        
        # Calculate average prices for the stable period
        avg_p1 = np.mean(stable_actions_a1)
        avg_p2 = np.mean(stable_actions_a2)
        
        # Calculate standard deviation of prices for the stable period
        std_p1 = np.std(stable_actions_a1)
        std_p2 = np.std(stable_actions_a2)
        
        # Compute profits based on average prices
        p = np.array([avg_p1, avg_p2])
        pi = game.compute_profits(p)  # Expected to return [pi1, pi2]
        
        if len(pi) != 2:
            raise ValueError(f"compute_profits should return a list or array of two elements, got {len(pi)} elements.")
        
        pi1, pi2 = pi
        
        # Compute Delta for each agent
        delta1 = (pi1 - pi1_N) / (pi1_M - pi1_N)
        delta2 = (pi2 - pi2_N) / (pi2_M - pi2_N)
        
        # Append the results to the lists
        list_avg_p1.append(avg_p1)
        list_avg_p2.append(avg_p2)
        list_std_p1.append(std_p1)
        list_std_p2.append(std_p2)
        list_pi1.append(pi1)
        list_pi2.append(pi2)
        list_delta1.append(delta1)
        list_delta2.append(delta2)

    # Calculate overall statistics for prices
    overall_avg_p1 = np.mean(list_avg_p1)
    overall_std_p1 = np.std(list_avg_p1)
    overall_avg_p2 = np.mean(list_avg_p2)
    overall_std_p2 = np.std(list_avg_p2)
    
    # Calculate overall statistics for profits
    overall_avg_pi1 = np.mean(list_pi1)
    overall_std_pi1 = np.std(list_pi1)
    overall_avg_pi2 = np.mean(list_pi2)
    overall_std_pi2 = np.std(list_pi2)
    
    # Calculate overall statistics for Delta
    overall_avg_delta1 = np.mean(list_delta1)
    overall_std_delta1 = np.std(list_delta1)
    overall_avg_delta2 = np.mean(list_delta2)
    overall_std_delta2 = np.std(list_delta2)
    
    # Print the results
    print(f"Price Statistics over {iterations} trajecotories:")
    print(f"Agent 1 - Average Price: {overall_avg_p1:.4f}, Standard Deviation: {overall_std_p1:.4f}")
    print(f"Agent 2 - Average Price: {overall_avg_p2:.4f}, Standard Deviation: {overall_std_p2:.4f}\n")
    
    print(f"Profit Statistics over {iterations} trajecotories:")
    print(f"Agent 1 - Average Profit: {overall_avg_pi1:.4f}, Standard Deviation: {overall_std_pi1:.4f}")
    print(f"Agent 2 - Average Profit: {overall_avg_pi2:.4f}, Standard Deviation: {overall_std_pi2:.4f}\n")
    
    print(f"Delta Statistics over {iterations} trajecotories:")
    print(f"Agent 1 - Average Delta: {overall_avg_delta1:.4f}, Standard Deviation of Delta: {overall_std_delta1:.4f}")
    print(f"Agent 2 - Average Delta: {overall_avg_delta2:.4f}, Standard Deviation of Delta: {overall_std_delta2:.4f}")
    
    # Return the statistics as a dictionary
    return {
        'avg_price1': overall_avg_p1,
        'std_price1': overall_std_p1,
        'avg_price2': overall_avg_p2,
        'std_price2': overall_std_p2,
        'avg_profit1': overall_avg_pi1,
        'std_profit1': overall_std_pi1,
        'avg_profit2': overall_avg_pi2,
        'std_profit2': overall_std_pi2,
        'avg_delta1': overall_avg_delta1,
        'std_delta1': overall_std_delta1,
        'avg_delta2': overall_avg_delta2,
        'std_delta2': overall_std_delta2
    }


def create_directed_network_graph(adj_matrix, node_labels):
    """
    Create and display a directed network graph based on an adjacency matrix.

    This function generates a directed graph using NetworkX from the provided 
    adjacency matrix. Nodes are labeled as specified, and the graph is visualized 
    using Matplotlib with a spring layout for better organization.

    Parameters
    ----------
    adj_matrix : numpy.ndarray
        Adjacency matrix representing the connections between nodes.
    node_labels : list of str
        List of labels for each node in the graph. The length must match the 
        number of nodes in the adjacency matrix.

    Raises
    ------
    ValueError
        If the number of labels does not match the number of nodes in the adjacency matrix.

    Returns
    -------
    None
        Displays the directed network graph plot.
    """
    if len(node_labels) != len(adj_matrix):
        raise ValueError("The number of labels must match the number of nodes in the matrix.")

    # Create a directed graph from the adjacency matrix
    G = nx.DiGraph(adj_matrix)

    # Relabel nodes with the provided labels
    mapping = {i: node_labels[i] for i in range(len(node_labels))}
    G = nx.relabel_nodes(G, mapping)

    # Set up the plot
    plt.figure(figsize=(10, 10))

    # Use a spring layout for better organization
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


def make_adjacency(Agent1, Agent2, Q1, Q2, labels='index', plot_graph=False):
    """
    Create an adjacency matrix representing interactions between two agents based on their Q-values.

    This function constructs an adjacency matrix where each node represents a joint action 
    of both agents. An edge from node X to node Y exists if the joint action Y is the 
    greedy response based on the Q-values of both agents when the current joint action is X.

    Parameters
    ----------
    Agent1 : object
        The first agent, which must have Q-values.
    Agent2 : object
        The second agent, which must have Q-values.
    Q1 : numpy.ndarray
        Q-values for Agent1, indexed by state-action pairs.
    Q2 : numpy.ndarray
        Q-values for Agent2, indexed by state-action pairs.
    labels : str, optional
        Labeling scheme for nodes. Options:
            - 'price': Labels nodes with their joint action prices.
            - 'index': Labels nodes with their joint action indices.
        Default is 'index'.
    plot_graph : bool, optional
        If True, creates and displays a directed network graph of the adjacency matrix.
        Default is False.

    Raises
    ------
    ValueError
        If either Agent1 or Agent2 does not use Q-values.

    Returns
    -------
    numpy.ndarray
        Adjacency matrix representing interactions between agents.
        Shape: (len1*len2, len1*len2)
    """
    # Check if both agents have Q-values
    def has_q_vals(agent):
        return (type(agent).__name__ in ['Q_Learning', 'Batch_SARSA', 'Dec_Q'] or
                any(base.__name__ in ['Q_Learning', 'Batch_SARSA', 'Dec_Q'] for base in type(agent).__bases__))

    if not has_q_vals(Agent1) or not has_q_vals(Agent2):
        raise ValueError("Both agents must have Q-values.")

    len1 = len(Agent1.a1_space)
    len2 = len(Agent2.a1_space)

    act1 = Agent1.a1_space
    act2 = Agent2.a1_space

    node_names = []
    if labels == 'price':
        for i in range(len1):
            for j in range(len2):
                node_names.append(f"({act1[i]},{act2[j]})")
    elif labels == 'index':
        for i in range(len1):
            for j in range(len2):
                node_names.append(f"({i},{j})")
    else:
        raise ValueError("labels must be either 'price' or 'index'.")

    adj_matrix = np.zeros((len1 * len2, len1 * len2))

    for i in range(len1):
        for j in range(len2):
            x = (i * len2 + j)

            price1 = act1[i]
            price2 = act2[j]

            idx1 = (Agent1.get_index_1(price1), Agent1.get_index_2(price2))
            idx2 = (Agent2.get_index_1(price2), Agent2.get_index_2(price1))
            
            try:
                y = (np.argmax(Q1[idx1]) * len2 + np.argmax(Q2[idx2]))
                adj_matrix[x, y] = 1
            except IndexError:
                # Handle cases where indices are out of bounds
                continue

    if plot_graph:
        create_directed_network_graph(adj_matrix, node_names)

    return adj_matrix


def check_rp(adj_matrix):
    """
    Check certain properties of the directed graph represented by the adjacency matrix.

    This function evaluates the graph to determine if it satisfies the following conditions:
    1. The Nash node is within the largest weakly connected component.
    2. There exists at least one limiting strongly connected component.
    3. The Nash node is not part of any limiting strongly connected component.

    Parameters
    ----------
    adj_matrix : numpy.ndarray
        Adjacency matrix representing the directed graph.

    Returns
    -------
    int
        Returns 1 if all conditions are satisfied, otherwise returns 0.
    """
    # Step 1: Create the directed graph from the adjacency matrix
    G = nx.DiGraph(adj_matrix)

    # Find all weakly connected components
    weakly_connected_components = list(nx.weakly_connected_components(G))

    if not weakly_connected_components:
        return 0  # No connected components

    # Identify the largest weakly connected component
    largest_wcc = max(weakly_connected_components, key=len)

    # Create a subgraph of the largest weakly connected component
    subG = G.subgraph(largest_wcc).copy()

    # Step 2: Find strongly connected components within the subgraph
    sccs = list(nx.strongly_connected_components(subG))

    # Identify limiting nodes or limiting cycles
    limiting_sccs = []
    for scc in sccs:
        is_limiting = True
        for node in scc:
            out_edges = subG.out_edges(node)
            for _, v in out_edges:
                if v not in scc:
                    is_limiting = False
                    break
            if not is_limiting:
                break
        if is_limiting:
            limiting_sccs.append(scc)

    # Check if there's at least one limiting node or cycle
    has_limiting_scc = len(limiting_sccs) > 0

    if not has_limiting_scc:
        return 0  # No limiting strongly connected components

    # Step 3: Find the Nash node (node with the maximum in-degree)
    in_degrees = dict(G.in_degree())
    max_in_degree = max(in_degrees.values())
    nash_nodes = [node for node, deg in in_degrees.items() if deg == max_in_degree]
    nash_node = nash_nodes[0]  # You can pick any if multiple nodes have the same in-degree

    # Check if the Nash node is in the largest weakly connected component
    nash_in_wcc = nash_node in largest_wcc

    # Step 4: Ensure the Nash node is not in any limiting SCC
    nash_in_limiting_scc = any(nash_node in scc for scc in limiting_sccs)

    # Step 5: Return 1 if all conditions are satisfied, else 0
    if nash_in_wcc and has_limiting_scc and not nash_in_limiting_scc:
        return 1
    else:
        return 0


def plot_rp(Agent1_list, Agent2_list, time_step=1):
    """
    Process Q-values from two agents, compute the averaged check_rp values per time step, and plot the results.

    This function evaluates the adjacency matrices derived from agents' Q-values at specified time steps,
    computes the `check_rp` metric for each, averages these metrics across all trajectories, and
    visualizes the average `check_rp` values over time.

    Parameters
    ----------
    Agent1_list : list of lists
        List where each sublist contains instances of Agent1 at different time steps in a simulation.
    Agent2_list : list of lists
        List where each sublist contains instances of Agent2 at different time steps in a simulation.
    Q1_list : list of lists
        List where each sublist contains Q-values for Agent1 at different time steps in a simulation.
    Q2_list : list of lists
        List where each sublist contains Q-values for Agent2 at different time steps in a simulation.
    time_step : int, optional
        The interval of time steps to consider for evaluation (default is 1).

    Returns
    -------
    list
        A list of averaged `check_rp` values per specified time step.
    """
    num_trajectories = len(Agent1_list)
    assert num_trajectories == len(Agent2_list), "Q1_list and Q2_list must have the same number of trajectories."

    # Get lengths of all trajectories to find the maximum length
    lengths = [len(traj) for traj in Agent1_list]
    max_length = max(lengths)

    # Initialize list to store check_rp values per time step
    check_rp_values_per_time_step = []

    # Time steps to consider based on the user-defined time_step interval
    time_steps = range(0, max_length, time_step)

    for t in time_steps:
        check_rp_values_at_t = []
        for x in range(num_trajectories):
            traj_length = lengths[x]

            # If the trajectory is shorter, use the last Q-value
            if t < traj_length:
                Agent1 = Agent1_list[x][t]
                Agent2 = Agent2_list[x][t]
                Q1 = Agent1.Q
                Q2 = Agent2.Q
            else:
                Agent1 = Agent1_list[x][-1]
                Agent2 = Agent2_list[x][-1]
                Q1 = Agent1.Q
                Q2 = Agent2.Q

            # Compute the adjacency matrix using the predefined function
            adj_matrix = make_adjacency(Agent1, Agent2, Q1, Q2)

            # Compute the check_rp value
            check_rp_value = check_rp(adj_matrix)

            # Append the check_rp value for this trajectory at time t
            check_rp_values_at_t.append(check_rp_value)

        # Compute the average check_rp value across all trajectories for time t
        avg_check_rp = sum(check_rp_values_at_t) / num_trajectories
        check_rp_values_per_time_step.append(avg_check_rp)

    # Plot the averaged check_rp values over time
    plt.figure(figsize=(10, 6))
    plt.plot([t for t in time_steps], check_rp_values_per_time_step, marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Average check_rp Value')
    plt.ylim((0,1))
    plt.title('Average check_rp Value over Time')
    plt.grid(True)
    plt.show()

    return check_rp_values_per_time_step


def profit_graph(game, a1_lists, a2_lists):
    """
    Plot the average profit over time for both players with confidence intervals.

    This function calculates the profits for each agent at every time step across all simulations,
    computes the mean and confidence intervals, and visualizes the average profits for both agents
    over time.

    Parameters
    ----------
    game : object
        The game environment, which must have a `compute_profits` method.
    Agent1 : object
        The first agent participating in the simulations.
    Agent2 : object
        The second agent participating in the simulations.
    a1_lists : list of lists
        List where each sublist contains the actions taken by Agent1 in a simulation.
    a2_lists : list of lists
        List where each sublist contains the actions taken by Agent2 in a simulation.

    Returns
    -------
    dict
        A dictionary containing:
            - 'time': numpy.ndarray of time steps.
            - 'mean_player1_profits': numpy.ndarray of mean profits for Player 1.
            - 'mean_player2_profits': numpy.ndarray of mean profits for Player 2.
            - 'lower_bound_player1': numpy.ndarray of lower confidence interval for Player 1.
            - 'upper_bound_player1': numpy.ndarray of upper confidence interval for Player 1.
            - 'lower_bound_player2': numpy.ndarray of lower confidence interval for Player 2.
            - 'upper_bound_player2': numpy.ndarray of upper confidence interval for Player 2.
    """
    # Ensure a1_lists and a2_lists have the same length
    if len(a1_lists) != len(a2_lists):
        raise ValueError("The number of a1_lists and a2_lists must be the same.")
    
    num_simulations = len(a1_lists)
    
    # Determine the maximum simulation length
    max_length = max(max(len(sim) for sim in a1_lists), max(len(sim) for sim in a2_lists))
    
    # Initialize arrays to store profits
    player1_profits = np.full((num_simulations, max_length), np.nan)
    player2_profits = np.full((num_simulations, max_length), np.nan)
    
    # Calculate profits for each simulation
    for i, (a1_list, a2_list) in enumerate(zip(a1_lists, a2_lists)):
        # Determine the length of the current simulation
        current_length = min(len(a1_list), len(a2_list))
        if current_length == 0:
            continue  # Skip if no actions
        
        profit_list = [game.compute_profits(np.array([p1, p2])) for p1, p2 in zip(a1_list[:current_length], a2_list[:current_length])]
        
        player1_profits[i, :current_length] = [profit[0] for profit in profit_list]
        player2_profits[i, :current_length] = [profit[1] for profit in profit_list]
    
    # Calculate mean profits, ignoring NaN values
    mean_player1_profits = np.nanmean(player1_profits, axis=0)
    mean_player2_profits = np.nanmean(player2_profits, axis=0)
    
    # Calculate standard deviation and standard error, ignoring NaN values
    std_player1_profits = np.nanstd(player1_profits, axis=0, ddof=1)
    std_player2_profits = np.nanstd(player2_profits, axis=0, ddof=1)
    
    n_player1 = np.sum(~np.isnan(player1_profits), axis=0)
    n_player2 = np.sum(~np.isnan(player2_profits), axis=0)
    
    # Avoid division by zero
    stderr_player1 = np.where(n_player1 > 1, std_player1_profits / np.sqrt(n_player1), np.nan)
    stderr_player2 = np.where(n_player2 > 1, std_player2_profits / np.sqrt(n_player2), np.nan)
    
    # Compute t-multiplier for confidence intervals
    confidence = 0.95
    df_player1 = n_player1 - 1
    df_player2 = n_player2 - 1
    
    # Handle degrees of freedom <=0
    t_multiplier_player1 = np.where(df_player1 > 0, stats.t.ppf((1 + confidence) / 2., df_player1), np.nan)
    t_multiplier_player2 = np.where(df_player2 > 0, stats.t.ppf((1 + confidence) / 2., df_player2), np.nan)
    
    # Compute confidence intervals
    lower_bound_player1 = mean_player1_profits - t_multiplier_player1 * stderr_player1
    upper_bound_player1 = mean_player1_profits + t_multiplier_player1 * stderr_player1
    
    lower_bound_player2 = mean_player2_profits - t_multiplier_player2 * stderr_player2
    upper_bound_player2 = mean_player2_profits + t_multiplier_player2 * stderr_player2
    
    # Create a time list
    time = np.arange(1, max_length + 1)
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot average profits with confidence intervals
    plt.plot(time, mean_player1_profits, label='Player 1')
    plt.fill_between(time, lower_bound_player1, upper_bound_player1, alpha=0.3)
    
    plt.plot(time, mean_player2_profits, label='Player 2')
    plt.fill_between(time, lower_bound_player2, upper_bound_player2, alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Average Profit')
    plt.title('Average Profit over Time for Both Players with Confidence Intervals')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.show()
    
    # Return the data for further analysis if needed
    return {
        'time': time,
        'mean_player1_profits': mean_player1_profits,
        'mean_player2_profits': mean_player2_profits,
        'lower_bound_player1': lower_bound_player1,
        'upper_bound_player1': upper_bound_player1,
        'lower_bound_player2': lower_bound_player2,
        'upper_bound_player2': upper_bound_player2
    }

def state_heatmap(game, Agent1, Agent2, a1_list, a2_list, ts):
    """
    Generate a heatmap of joint state distributions.

    Parameters:
    ----------
    game : object
        The game environment.
    Agent1 : object
        The first agent.
    Agent2 : object
        The second agent.
    a1_list : list of lists
        List of actions taken by Agent1 across simulations.
    a2_list : list of lists
        List of actions taken by Agent2 across simulations.

    Returns:
    -------
    joint_state_counts : numpy.ndarray
        2D array of joint state counts.
    """

    # Initialize joint state counts matrix
    num_a1_actions = len(Agent1.a1_space)
    num_a2_actions = len(Agent2.a1_space)
    joint_state_counts = np.zeros((num_a1_actions, num_a2_actions))

    # Create mappings from action values to indices in the action spaces
    a1_action_to_index = {action: idx for idx, action in enumerate(Agent1.a1_space)}
    a2_action_to_index = {action: idx for idx, action in enumerate(Agent2.a1_space)}

    # Count joint states in the stable period of each simulation
    for sim_idx in range(len(a1_list)):
        a1_actions = a1_list[sim_idx]
        a2_actions = a2_list[sim_idx]
        sim_length = len(a1_actions)
        # Determine the starting index for the stable period
        start_idx = max(0, sim_length - int(game.tstable/ts))
        for step_idx in range(start_idx, sim_length):
            a1_action = a1_actions[step_idx]
            a2_action = a2_actions[step_idx]

            # Map actions to indices in the action spaces
            # Handle cases where the action might not be in the agent's action space
            a1_index = a1_action_to_index.get(a1_action, None)
            a2_index = a2_action_to_index.get(a2_action, None)

            # If the action is not found, find the closest action in the action space
            if a1_index is None:
                closest_a1_action = Agent1.a1_space[np.argmin(np.abs(Agent1.a1_space - a1_action))]
                a1_index = a1_action_to_index[closest_a1_action]

            if a2_index is None:
                closest_a2_action = Agent2.a1_space[np.argmin(np.abs(Agent2.a1_space - a2_action))]
                a2_index = a2_action_to_index[closest_a2_action]

            # Increment the count for this joint action
            joint_state_counts[a1_index, a2_index] += 1

    # Normalize counts to probabilities
    total_counts = np.sum(joint_state_counts)
    joint_state_probs = joint_state_counts / total_counts if total_counts > 0 else joint_state_counts

    # Prepare labels for the heatmap
    a1_labels = [f"{action:.2f}" for action in Agent1.a1_space]
    a2_labels = [f"{action:.2f}" for action in Agent2.a1_space]

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        joint_state_probs,
        annot=True,
        fmt=".2f",
        xticklabels=a2_labels,
        yticklabels=a1_labels,
        cmap="YlGnBu"
    )
    plt.xlabel("Agent2 Actions")
    plt.ylabel("Agent1 Actions")
    plt.title("Joint State Distribution Heatmap")
    plt.show()

    return joint_state_counts  # Return counts if needed


### VERY BUGGY, To Be fixed later

# def simulate_deviation(game, Agent1, Agent2, tdeviate, tmax, deviated_price=0, deviated_index=0, index=True):
#     """
#     Simulate a deviation in Agent1's strategy and plot the resulting actions over time.

#     This function forces Agent1 to set a specific price either by index or by explicit value
#     for the first `tdeviate` time steps. After the deviation period, Agent1 resumes its
#     normal strategy. The actions of both agents are recorded and plotted over the simulation period.

#     Parameters
#     ----------
#     game : object
#         The game environment, which must have a `compute_profits` method.
#     Agent1 : object
#         The first agent, whose actions will be deviated.
#     Agent2 : object
#         The second agent participating in the simulation.
#     tdeviate : int
#         The number of initial time steps during which Agent1's action is deviated.
#     tmax : int
#         The total number of time steps to simulate.
#     deviated_price : float, optional
#         The specific price value Agent1 should set during the deviation period. Used if `index` is False.
#         Default is 0.
#     deviated_index : int, optional
#         The index of the price in Agent1's action space to set during the deviation period.
#         Used if `index` is True. Default is 0.
#     index : bool, optional
#         Determines whether to use the `deviated_index` or `deviated_price` for deviation.
#         If True, uses `deviated_index`; otherwise, uses `deviated_price`. Default is True.

#     Returns
#     -------
#     None
#         Displays a plot of Agent1 and Agent2's actions over time.
#     """
#     if index:
#         fixed_price = Agent1.a1_space[deviated_index]
#     else:
#         fixed_price = deviated_price

#     all_actions = []
#     a1_values = []
#     a2_values = []

#     for t in range(int(tmax)):
#         tbig = 1000000

#         s = (Agent1.final_price, Agent2.final_price)

#         if t <= tdeviate:
#             a1 = fixed_price  
#         else:
#             a1 = Agent1.pick_strategies(game, s, tbig)

#         a2 = Agent2.pick_strategies(game, s[::-1], tbig)

#         a = (a1, a2)
#         a_prof = np.array([a1, a2])
#         all_actions.append(a)
#         a1_values.append(a1)
#         a2_values.append(a2)
#         pi1 = game.compute_profits(a_prof)
#         s = a

#     print(a1_values)
#     print(a2_values)
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(a1_values)), a1_values, label='Agent 1')
#     plt.plot(range(len(a2_values)), a2_values, label='Agent 2')
#     # plt.ylim((0,game.k-1))
#     plt.xlabel('Time')
#     plt.ylabel('Action Value')
#     plt.title('Agent Actions Over Time')
#     plt.legend()
#     plt.grid(True)
#     plt.show()



