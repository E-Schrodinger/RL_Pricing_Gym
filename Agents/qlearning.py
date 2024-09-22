"""
Q-learning Functions

This module implements Q-learning algorithms for reinforcement learning in game theory contexts.
"""

import sys
import numpy as np
import copy


class Q_Learning:
    """
    A class implementing Q-learning for reinforcement learning in games.

    This class provides methods for initializing, updating, and using a Q-function
    to make decisions in a game-theoretic context.

    Attributes:
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

        Parameters:
        ----------
        game : object
            The game environment.
        **kwargs : dict
            Additional parameters to override default values.
        """
        self.delta = kwargs.get('delta', 0.95)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.beta = kwargs.get('beta', 4e-6)

        self.Q = self.init_Q(game)

    def init_Q(self, game):
        """
        Initialize the Q-function.

        This method creates and initializes the Q-function based on the game's
        dimensions and initial payoffs.

        Parameters:
        ----------
        game : object
            The game environment.

        Returns:
        -------
        ndarray
            Initialized Q-function.
        """
        Q = np.zeros((game.n,) + game.sdim + (game.k,))
        for n in range(game.n):
            # Calculate mean payoffs across opponent's actions
            pi = np.mean(game.PI[:, :, n], axis=1 - n)
            # Initialize Q-values with discounted mean payoffs
            Q[n] = np.tile(pi, game.sdim + (1,)) / (1 - self.delta)
        return Q
    
    def reset(self, game):
        """
        Reset the Q-function to its initial state.

        Parameters:
        ----------
        game : object
            The game environment.
        """
        self.Q = self.init_Q(game)
    
    def pick_strategies(self, game, s, t):
        """
        Choose actions based on the current Q-function and exploration strategy.

        This method implements an epsilon-greedy strategy with decaying exploration rate.

        Parameters:
        ----------
        game : object
            The game environment.
        s : tuple
            Current state.
        t : int
            Current time step.

        Returns:
        -------
        ndarray
            Chosen actions for each player.
        """
        a = np.zeros(game.n).astype(int)
        # Calculate exploration probability with exponential decay
        pr_explore = np.exp(- t * self.beta)
        # pr_explore = 0.1  # Alternatively, use a fixed exploration rate
        
        # Determine whether to explore or exploit for each player
        e = (pr_explore > np.random.rand(game.n))
        
        for n in range(1):  # Note: This loop only runs once. Consider removing the loop if n is always 0.
            if e[n]:
                # Explore: choose a random action
                a[n] = np.random.randint(0, game.k)
            else:
                # Exploit: choose the action with the highest Q-value
                a[n] = np.argmax(self.Q[(n,) + tuple(s)])
        return a
    
    def update_function(self, game, s, a, s1, pi, stable, t, tol=1e-3):
        """
        Update the Q-function based on the observed transition and reward.

        This method implements the Q-learning update rule and checks for convergence.

        Parameters:
        ----------
        game : object
            The game environment.
        s : tuple
            Current state.
        a : tuple
            Chosen actions.
        s1 : tuple
            Next state.
        pi : ndarray
            Observed payoffs.
        stable : int
            Number of consecutive stable updates.
        t : int
            Current time step.
        tol : float, optional
            Tolerance for considering Q-values as converged (default: 1e-1).

        Returns:
        -------
        tuple
            Updated Q-function and stability counter.
        """
        for n in range(1):  # Note: This loop only runs once. Consider removing the loop if n is always 0.
            # Construct the index for the current state-action pair
            subj_state = (n,) + tuple(s) + (a[n],)
            
            # Store old Q-values for stability check
            old_q = self.Q[0].copy()
            old_value = self.Q[subj_state]
            
            # Compute the maximum Q-value for the next state
            max_q1 = np.max(self.Q[(n,) + tuple(s1)])
            
            # Compute the new Q-value using the Q-learning update rule
            new_value = pi[n] + self.delta * max_q1
            
            # Update the Q-value using a weighted average of old and new values
            self.Q[subj_state] = (1 - game.alpha) * old_value + game.alpha * new_value
            
            # Check for stability (convergence)
            same_q = np.allclose(old_q, self.Q[0], tol)
            stable = (stable + same_q) * same_q  # Reset to 0 if not stable, increment if stable
        
        return self.Q, stable