    
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from Metrics.Simulation_Base import Simulations
from scipy import stats

def find_regret(game, Agent1, Agent2, a1_lists, a2_lists, use_loglog=False, regret_type="ratio"):
    """
    Calculate the average regret over time with confidence intervals for multiple action lists.

    Parameters:
    - game: The game environment.
    - Agent1: The first agent.
    - Agent2: The second agent.
    - a1_lists: List of lists containing actions for Agent1.
    - a2_lists: List of lists containing actions for Agent2.
    - use_loglog: Whether to plot using a log-log scale.
    - regret_type: Type of regret calculation ("ratio" or "subtract").

    Returns:
    - mean_regret: The average regret over time.
    - lower_bound: The lower bound of the confidence interval.
    - upper_bound: The upper bound of the confidence interval.
    """
    if len(a1_lists) != len(a2_lists):
        raise ValueError("The number of a1_lists and a2_lists must be the same.")

    all_regrets = []

    for idx, (a1_list, a2_list) in enumerate(zip(a1_lists, a2_lists)):
        # Determine the length of the current pair
        current_length = min(len(a1_list), len(a2_list))
        if current_length == 0:
            continue  # Skip if no actions

        # Calculate total rewards for each action
        a1_space = len(Agent1.a1_prices)
        a2_space = len(Agent2.a1_prices)

        total_rewards = np.array([
            sum(game.compute_profits(np.array([Agent1.a1_prices[i], a2_list[j]]))[0] for j in range(len(a2_list)))
            for i in range(a1_space)
        ])

        best_strategy = np.argmax(total_rewards)

        # Calculate cumulative sums
        best_rewards = np.cumsum([
            game.compute_profits(np.array([Agent1.a1_prices[best_strategy], a2_list[j]]))[0]
            for j in range(current_length)
        ])

        actual_rewards = np.cumsum([
            game.compute_profits(np.array([a1_list[j], a2_list[j]]))[0]
            for j in range(current_length)
        ])

        # Calculate regret
        if regret_type == "subtract":
            regret_over_time = best_rewards - actual_rewards
        else:
            # To avoid division by zero, replace zero best_rewards with np.nan
            # best_rewards = np.where(best_rewards == 0, np.nan, best_rewards)
            regret_over_time = 1 - (actual_rewards / best_rewards)

        regret_over_time = regret_over_time.tolist()
        all_regrets.append(regret_over_time)

    if not all_regrets:
        raise ValueError("No valid action lists provided.")

    # Determine the maximum length among all regret lists
    max_length = max(len(regret) for regret in all_regrets)

    # Initialize a 2D array without filling (will populate in the loop)
    regret_matrix = np.empty((len(all_regrets), max_length))

    # Populate the matrix with regret values and fill remaining with last value
    for i, regret in enumerate(all_regrets):
        current_length = len(regret)
        regret_matrix[i, :current_length] = regret
        if current_length < max_length:
            # Fill the remaining entries with the last value of the regret list
            regret_matrix[i, current_length:] = regret[-1]

    # # Initialize a 2D array with NaNs
    # regret_matrix = np.full((len(all_regrets), max_length), np.nan)

    # # Populate the matrix with regret values
    # for i, regret in enumerate(all_regrets):
    #     regret_matrix[i, :len(regret)] = regret

    # Calculate mean and confidence intervals, ignoring NaNs
    mean_regret = np.nanmean(regret_matrix, axis=0)
    std_regret = np.nanstd(regret_matrix, axis=0)
    n = np.sum(~np.isnan(regret_matrix), axis=0)
    confidence = 0.95
    stderr = std_regret / np.sqrt(n)
    t_multiplier = stats.t.ppf((1 + confidence) / 2., n-1)
    lower_bound = mean_regret - t_multiplier * stderr
    upper_bound = mean_regret + t_multiplier * stderr

    plot_average_regret(mean_regret, lower_bound, upper_bound, use_loglog)

    return mean_regret, lower_bound, upper_bound

def plot_average_regret(mean_regret, lower_bound, upper_bound, use_loglog=False):
    """
    Plot the average regret over time with confidence intervals.

    Parameters:
    - mean_regret: The average regret over time.
    - lower_bound: The lower bound of the confidence interval.
    - upper_bound: The upper bound of the confidence interval.
    - use_loglog: Whether to plot using a log-log scale.
    """
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(1, len(mean_regret) + 1)

    if use_loglog:
        plt.loglog(time_steps, mean_regret, label='Average Regret')
        plt.fill_between(time_steps, lower_bound, upper_bound, alpha=0.3, label='95% Confidence Interval')
        plt.xlabel('Time steps (log scale)')
        plt.ylabel('Regret (log scale)')
        plt.title('Average Regret over Time with Confidence Intervals (Log-Log Plot)')
    else:
        plt.plot(time_steps, mean_regret, label='Average Regret')
        plt.fill_between(time_steps, lower_bound, upper_bound, alpha=0.3, label='95% Confidence Interval')
        plt.xlabel('Time steps')
        plt.ylabel('Regret')
        plt.title('Average Regret over Time with Confidence Intervals')

    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()
    

