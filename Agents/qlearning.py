"""
Q-learning Functions

This module implements Q-learning algorithms for reinforcement learning in game theory contexts.
"""

import sys
import numpy as np
import copy
from Agents.QBase import QBase


class Q_Learning(QBase):
    """
    A class implementing Q-learning for reinforcement learning in games.

    This class provides methods for initializing, updating, and using a Q-function
    to make decisions in a game-theoretic context.

    Attributes
    ----------
    delta : float
        Discount factor for future rewards (default: 0.95).
    epsilon : float
        Exploration rate for epsilon-greedy strategy (default: 0.1).
    beta : float
        Decay rate for exploration probability (default: 4e-6).
    Q : ndarray
        Q-function storing action-value estimates.
    """

    def __init__(self, game, **kwargs):
        """
        Initialize the Q-learning agent.

        Parameters
        ----------
        game : object
            The game environment.
        **kwargs : dict
            Additional parameters to override default values.
        """
        
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.beta = kwargs.get('beta', 4e-6)

        super().__init__(game, **kwargs)


    def reset(self, game):
        """
        Reset the Q-function to its initial state.

        Parameters
        ----------
        game : object
            The game environment.
        """
        self.Q = self.make_Q()
        self.state_space = copy.copy(self.a1_space)
    
    def pick_strategies(self, game, p, t):
        """
        Choose actions based on the current Q-function and exploration strategy.

        This method implements an epsilon-greedy strategy with a decaying exploration rate.

        Parameters
        ----------
        game : object
            The game environment.
        p : tuple
            Current players' profiles or actions.
        t : int
            Current time step.

        Returns
        -------
        int
            Chosen action for the player.
        """
        s = (self.get_index_1(p[0]), self.get_index_2(p[1]))
        a = np.zeros(1)
        # Calculate exploration probability with exponential decay
        pr_explore = np.exp(- t * self.beta)
        # pr_explore = 0.1  # Alternatively, use a fixed exploration rate
        
        # Determine whether to explore or exploit
        e = (pr_explore > np.random.rand())

        if e:
            # Explore: choose a random action
            a = np.random.randint(0, self.k)
        else:
            # Exploit: choose the action with the highest Q-value
            a = np.argmax(self.Q[tuple(s)])
    
        self.a_price = self.a1_space[a]
        return self.a_price
    
    def update_function(self, game, p, a_prices, pi, stable, t, tol=1e-5):
        """
        Update the Q-function based on the observed transition and reward.

        This method implements the Q-learning update rule and checks for convergence.

        Parameters
        ----------
        game : object
            The game environment.
        p : tuple
            Current players' profiles or actions.
        a_prices : tuple
            Chosen action prices for the players.
        pi : ndarray
            Observed payoffs.
        stable : int
            Number of consecutive stable updates.
        t : int
            Current time step.
        tol : float, optional
            Tolerance for considering Q-values as converged (default: 1e-5).

        Returns
        -------
        tuple
            Updated Q-function and stability counter.
        """
        self.dt = t
        s = (self.get_index_1(p[0]), self.get_index_2(p[1]))
        a = (self.get_index_1(a_prices[0]), self.get_index_2(a_prices[1]))

        # Construct the index for the current state-action pair
        subj_state = tuple(s) + (a[0],)
        # Store old Q-values for stability check
        old_q = self.Q.copy()
        old_value = self.Q[subj_state]
        # Compute the maximum Q-value for the next state
        
        
        max_q1 = np.max(self.Q[tuple(a)])
    
        
        # Compute the new Q-value using the Q-learning update rule
        new_value = pi + self.delta * max_q1

        
        # Update the Q-value using a weighted average of old and new values
        self.Q[subj_state] = (1 - game.alpha) * old_value + game.alpha * new_value
        
        # Check for stability (convergence)
        same_q = np.allclose(old_q, self.Q, atol=tol)
        stable = (stable + same_q) * same_q  # Reset to 0 if not stable, increment if stable
    
        return self.Q, stable