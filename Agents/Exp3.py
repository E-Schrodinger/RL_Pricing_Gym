"""
EXP3 Algorithm Implementation with Decaying Exploration Rate

This module implements the EXP3 (Exponential-weight algorithm for Exploration and Exploitation)
algorithm for reinforcement learning in game theory contexts, incorporating a decaying
exploration rate over time.
"""

import sys
import numpy as np
import copy
from Agents.QBase import QBase


class Exp3(QBase):
    """
    A class implementing the EXP3 algorithm with decaying exploration rate for reinforcement learning in games.

    This class provides methods for initializing, updating, and using a probability distribution
    over actions to make decisions in a game-theoretic context using the EXP3 strategy with
    exploration probability decaying as e^(-beta * t).

    Attributes:
    ----------
    beta : float
        Decay rate for the exploration probability (default: 4e-6).
    eta : float
        Learning rate for updating weights (default: 0.15).
    weights : ndarray
        Weight vector for each action.
    probabilities : ndarray
        Probability distribution over actions based on current weights.
    previous_weights : ndarray
        Previous weight vector for stability checking.
    """

    def __init__(self, game, **kwargs):
        """
        Initialize the EXP3 agent with decaying exploration rate.

        Parameters:
        ----------
        game : object
            The game environment.
        **kwargs : dict
            Additional parameters to override default values. Supports 'beta' and 'eta'.
        """
        self.beta = kwargs.get('beta', 4e-6)    # Decay rate for exploration
        self.eta = kwargs.get('eta', 0.15)      # Learning rate
        self.gamma = kwargs.get('gamma', 0.05)

        super().__init__(game, **kwargs)

        # Initialize weights uniformly
        self.weights = np.ones(self.k)
        self.probabilities = self.weights / np.sum(self.weights)
        

    def reset(self, game):
        """
        Reset the EXP3 weights to their initial state.

        Parameters:
        ----------
        game : object
            The game environment.
        """
        self.weights = np.ones(self.k)
        self.probabilities = self.weights / np.sum(self.weights)
        self.price_state_space = copy.copy(self.a1_space)
        self.previous_weights = self.weights.copy()

    def pick_strategies(self, game, p, t):
        """
        Choose actions based on the current probability distribution and decaying exploration rate.

        This method implements the EXP3 strategy with exploration probability decreasing as e^(-beta * t).

        Parameters:
        ----------
        game : object
            The game environment.
        p : ndarray
            Current players' strategies or states.
        t : int
            Current time step.

        Returns:
        -------
        ndarray
            Chosen actions for each player.
        """
        # Compute the current exploration probability with exponential decay
        gamma_t = self.gamma

        # Update probabilities based on current weights and decaying gamma
        self.probabilities = (1 - gamma_t) * (self.weights / np.sum(self.weights)) + (gamma_t / self.k)

        # Select an action based on the probability distribution
        action = np.random.choice(self.k, p=self.probabilities)

        self.a_price = self.a1_space[action]
        return self.a_price

    def update_function(self, game, p, a_prices, pi, stable, t, tol=1e-3):
        """
        Update the weights based on the observed reward.

        This method implements the EXP3 update rule.

        Parameters:
        ----------
        game : object
            The game environment.
        p : ndarray
            Current players' strategies or states.
        a_prices : ndarray
            Actions chosen by the players.
        pi : float
            Observed payoff or reward for the chosen action.
        stable : int
            Number of consecutive stable updates.
        t : int
            Current time step.
        tol : float, optional
            Tolerance for considering weights as converged (default: 1e-5).

        Returns:
        -------
        tuple
            Updated weights and stability counter.
        """
        # Identify the chosen action index
        chosen_action = self.get_index_1(a_prices[0])

        reward = pi  # Assuming pi corresponds to the reward for the chosen action

        # Compute the estimated reward
        estimated_reward = reward / self.probabilities[chosen_action]

        # Update the weight for the chosen action using the EXP3 update rule
        growth_factor = np.exp(self.eta * estimated_reward / self.k)
        self.weights[chosen_action] *= growth_factor

        # Normalize weights to prevent overflow/underflow
        self.weights = np.maximum(self.weights, 1e-10)  # Prevent weights from becoming too small
        self.weights /= np.sum(self.weights)

        # Check for stability (weights have converged)
        if t > 1:
            change = np.allclose(self.previous_weights, self.weights, atol=tol)
            if change < tol:
                stable += 1
            else:
                stable = 0
        self.previous_weights = self.weights.copy()

        return self.weights, stable

    def make_Q(self):
        """
        Create an initial weight vector for actions.

        Returns:
        -------
        ndarray
            Initialized weights.
        """
        return np.ones(self.k)

    def setup(self):
        """
        Setup method to initialize any additional attributes required before training.

        This method should be called after initialization.
        """
        self.previous_weights = self.weights.copy()