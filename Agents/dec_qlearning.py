"""
Batch SARSA Functions

This module implements Batch SARSA (State-Action-Reward-State-Action) algorithms 
for reinforcement learning in game theory contexts.
"""

import sys
import numpy as np
import copy
from Agents.QBase import QBase

class Dec_Q(QBase):
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
 
        super().__init__(game, **kwargs)
        
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.beta = kwargs.get('beta', 4e-6)
        self.batch_size = kwargs.get('batch_size', 1000)
        self.lamb = kwargs.get('lamb', 0.1)


        


  
    
  
    def reset(self, game):
        """Reset all data structures to initial state"""
        self.price_state_space = copy.copy(self.a1_space)
        self.Q = self.make_Q()
        self.Q_val = self.Q.copy()
        self.num = self.make_num()
        

    def pick_strategies(self, game, p, t):
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
        s = (self.get_index_1(p[0]), self.get_index_2(p[1]))
        a = np.zeros(1)
        # Calculate exploration probability with exponential decay
        pr_explore = np.exp(- t * self.beta)
        # pr_explore = 0.1  # Alternatively, use a fixed exploration rate
        
        # Determine whether to explore or exploit for each player
        e = (pr_explore > np.random.rand())
        
        if e:
            # Explore: choose a random action
            a = np.random.randint(0, self.k)
        else:
            # Exploit: choose the action with the highest Q-value
            a = np.argmax(self.Q[tuple(s)])
    
        a_price = self.a1_space[a]
        return a_price
    
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
        
    
        optimal = np.argmax(self.Q_val[tuple(s)])
        if a == optimal:
            probabilities = self.epsilon/self.Q_val.shape[0] + 1 - self.epsilon
        else:
            probabilities = self.epsilon/self.Q_val.shape[0]
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
        state = tuple(s_hat) + (a_hat,)
        self.Q[state] = self.Q_val[state]

    def update_function(self, game, p, a_prices, pi, stable, t, tol=1e-5):
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
        self.dt = t
        s = (self.get_index_1(p[0]), self.get_index_2(p[1]))
        a = (self.get_index_1(a_prices[0]), self.get_index_2(a_prices[1]))
       
        subj_state = tuple(s) + (a[0],)
        # print(subj_state)
        # print(self.Q_val.shape)
        old_value = self.Q_val[subj_state]
        
        # Update counters and accumulated rewards
        self.num[tuple(a)] += 1

        # Calculate learning rate
        a_t = 1/(self.num[tuple(a)]+1)
        
        # Calculate expected Q-value of next state
        Q_merge = 0
        for i in range(self.Q_val.shape[0]):
            Q_merge += self.X_function(game, a, i) * self.Q_val[ tuple(a) + (i,)]

        # Update Q-value
        self.Q_val[subj_state] = (1-a_t)*old_value + a_t*(pi + self.delta*Q_merge)

            

        # Perform batch update if necessary
        if (t % self.batch_size == 0):
            for s_hat in np.ndindex((self.Q_val.shape[0], self.Q_val.shape[1])):
                for a_hat in np.ndindex(self.Q_val.shape[2]):
                    old_q = self.Q[0].copy()
                    if np.random.uniform() >= self.lamb:
                        self.adaption_phase(game, s_hat, a_hat)
                    same_q = np.allclose(old_q, self.Q[0], tol)
                    stable = (stable + same_q) * same_q
            self.num.fill(0)
 
       
        return self.Q, stable