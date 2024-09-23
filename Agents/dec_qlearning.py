"""
Batch SARSA Functions

This module implements Batch SARSA (State-Action-Reward-State-Action) algorithms 
for reinforcement learning in game theory contexts.
"""

import sys
import numpy as np

class Dec_Q:
    """
    A class implementing Decentralized Q Learning for reinforcement learning in games.

    This class provides methods for initializing, updating, and using a Q-function
    with batch updates for SARSA learning in a game-theoretic context.

    Attributes:
    ----------
    delta : float
        Discount factor for future rewards (default: 0.95).
    epsilon : float
        Exploration rate for epsilon-greedy strategy (default: 0.1).
    beta : float
        Decay rate for exploration probability (default: 4e-6).
    batch_size : int
        Number of steps between batch updates (default: 1000).
    Q : ndarray
        Q-function storing action-value estimates.
    Q_val : ndarray
        Copy of Q-function used for value updates.
    trans : ndarray
        Transition function counting state transitions.
    num : ndarray
        Counter for state-action visits.
    reward : ndarray
        Accumulated rewards for each state-action pair.
    X : ndarray
        Probability distribution for action selection.
    """

    def __init__(self, game, **kwargs):
        """
        Initialize the Batch SARSA agent.

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
        self.batch_size = kwargs.get('batch_size', 1000)
        self.Qinit = kwargs.get('Qinit', 'uniform')
        self.lamb = kwargs.get('lamb', 0.1)

        self.Q = self.init_Q_act(game)
        self.Q_val = self.Q.copy()
        self.num = self.init_num(game)

        # Initialize action selection probabilities
        self.X = np.full(self.Q.shape, self.epsilon / game.k)
        for n in range(1):
            optimal_actions = np.argmax(self.Q[n], axis=-1)
            self.X[n][np.arange(game.sdim[0]), np.arange(game.sdim[1]), optimal_actions] += 1 - self.epsilon

    def init_Q_act(self, game):
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
        if self.Qinit == 'uniform':
            Q = np.random.rand(game.n, *game.sdim, game.k)
        elif self.Qinit == 'zero':
            Q = np.zeros((game.n,) + game.sdim + (game.k,))
        else:
            Q = np.zeros((game.n,) + game.sdim + (game.k,))
            for n in range(game.n):
                # Calculate mean payoffs across opponent's actions
                pi = np.mean(game.PI[:, :, n], axis=1 - n)
                # Initialize Q-values with discounted mean payoffs
                Q[n] = np.tile(pi, game.sdim + (1,)) / (1 - self.delta)
        return Q
    
    def init_num(self, game):
        """Initialize visit counter (n x #s x k)"""
        return np.zeros((game.n,) + game.sdim + (game.k,))
    
    def reset(self, game):
        """Reset all data structures to initial state"""
        self.Q = self.init_Q_act(game)
        self.Q_val = self.Q.copy()
        self.num = self.init_num(game)

    def pick_strategies(self, game, s, t):
        """
        Choose actions based on the current Q-function and exploration strategy.

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
        pr_explore = np.exp(- t * self.beta)
        e = (pr_explore > np.random.rand(game.n))
        for n in range(1):
            if e[n]:
                a[n] = np.random.randint(0, game.k)
            else:
                a[n] = np.argmax(self.Q[(n,) + tuple(s)])
        return a
    
    def X_function(self, game, s, a):
        """
        Calculate action selection probabilities.

        Parameters:
        ----------
        game : object
            The game environment.
        s : tuple
            Current state.
        a : int
            Action to calculate probability for.

        Returns:
        -------
        ndarray
            Probabilities of selecting action a for each player.
        """
        probabilities = np.zeros(game.n)
        for n in range(game.n):
            optimal = np.argmax(self.Q_val[(n,) + tuple(s)])
            if a == optimal:
                probabilities[n] = self.epsilon/game.k + 1 - self.epsilon
            else:
                probabilities[n] = self.epsilon/game.k
        return probabilities
    
    def adaption_phase(self, game, s_hat, a_hat):
        """
        Perform the adaptation phase of the Batch SARSA algorithm.

        Parameters:
        ----------
        game : object
            The game environment.
        s_hat : tuple
            Current state.
        a_hat : tuple
            Chosen actions.
        s_prime : tuple
            Next state.
        """
        for n in range(1):
            state = (n,) + tuple(s_hat) + (a_hat[n],)
            self.Q[state] = self.Q_val[state]
    
    def update_function(self, game, s, a, s1, pi, stable, t, tol = 1e-1):
        """
        Update the Q-function based on the observed transition and reward.

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
        for n in range(1):
            subj_state = (n,) + tuple(s) + (a[n],)
            old_value = self.Q_val[subj_state]
            
            # Update counters and accumulated rewards
            self.num[subj_state] += 1

            # Calculate learning rate
            a_t = 1/(self.num[subj_state]+1)
            
            # Calculate expected Q-value of next state
            Q_merge = 0
            for i in range(game.k):
                Q_merge += self.X_function(game, s1, i) * self.Q_val[(n,) + tuple(s1) + (i,)]

            # Update Q-value
            self.Q_val[subj_state] = (1-a_t)*old_value + a_t*(pi[n] + self.delta*Q_merge[n])

            # Check stability

        # Perform batch update if necessary
        if (t % self.batch_size == 0):
            for s_hat in np.ndindex(game.sdim):
                for a_hat in np.ndindex((game.k,) * game.n):
                    old_q = self.Q[0].copy()
                    if np.random.uniform() >= self.lamb:
                        self.adaption_phase(game, s_hat, a_hat)
                    same_q = np.allclose(old_q, self.Q[0], tol)
                    stable = (stable + same_q) * same_q
            self.num.fill(0)
 
       
        return self.Q, stable