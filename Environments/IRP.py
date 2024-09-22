"""
Model of algorithms and competition
"""

import numpy as np
from itertools import product
from scipy.optimize import fsolve
import sys


class IRP(object):
    """
    model

    Attributes
    ----------
    n : int
        number of players
    alpha : float
        product differentiation parameter
    mu : float
        product differentiation parameter
    a : int
        value of the products
    a0 : float
        value of the outside option
    c : float
        marginal cost
    k : int
        dimension of the grid
    tstable: int
        periods of game stability
    tmax: int
        maximum iterations of play
    dem_function : str or callable
        Demand function to use ('default' or custom function).
    sdim : tuple
        Dimensions of the state space.
    s0 : ndarray
        Initial state.
    p_minmax : tuple
        Competitive and monopoly prices.
    A : ndarray
        Discrete action space (possible prices).
    PI : ndarray
        Profit matrix for all possible states and actions.
    """

    def __init__(self, dem_function = 'default',**kwargs):
        """
        Initialize the IRP model with given or default parameters.

        Parameters:
        ----------
        dem_function : str or callable, optional
            Demand function to use (default is 'default').
        **kwargs : dict
            Additional parameters to override default values.
        """
        # Default properties
        self.n = kwargs.get('n', 2)
        self.alpha = kwargs.get('alpha', 0.15)
        self.c = kwargs.get('c', 1)
        self.a = kwargs.get('a', 2)
        self.a0 = kwargs.get('a0', 0)
        self.mu = kwargs.get('mu', 0.25)
        self.k = kwargs.get('k', 15)
        self.tstable = kwargs.get('tstable', 1e2)
        self.tmax = kwargs.get('tmax', 1e4)

        self.dem_function = dem_function

        # Derived properties
        self.sdim, self.s0 = self.init_state()
        self.p_minmax = self.compute_p_competitive_monopoly()
        self.A = self.init_actions()
        self.PI = self.init_PI()

        

    def demand(self, p):
        """
        Compute the demand for each firm given their prices.

        Parameters:
        ----------
        p : ndarray
            Array of prices set by each firm.

        Returns:
        -------
        ndarray
            Array of demand quantities for each firm.
        """
        e = np.exp((self.a - p) / self.mu)
        d = e / (np.sum(e) + np.exp(self.a0 / self.mu))
        return d

    def foc(self, p):
        """
        Compute the first-order condition for profit maximization.

        Parameters:
        ----------
        p : ndarray
            Array of prices set by each firm.

        Returns:
        -------
        ndarray
            Array of first-order condition values.
        """
        if self.dem_function == 'default':
            d = self.demand(p)
        else:
            d = self.dem_function(p)
        zero = 1 - (p - self.c) * (1 - d) / self.mu
        return np.squeeze(zero)

    def foc_monopoly(self, p):
        """
        Compute the first-order condition for a monopolist.

        Parameters:
        ----------
        p : ndarray
            Array of prices set by the monopolist for each product.

        Returns:
        -------
        ndarray
            Array of first-order condition values for a monopolist.
        """
        if self.dem_function == 'default':
            d = self.demand(p)
        else:
            d = self.dem_function(p)
        d1 = np.flip(d)
        p1 = np.flip(p)
        zero = 1 - (p - self.c) * (1 - d) / self.mu + (p1 - self.c) * d1 / self.mu
        return np.squeeze(zero)

    def compute_p_competitive_monopoly(self):
        """
        Compute competitive and monopoly prices.

        Returns:
        -------
        tuple
            A tuple containing competitive and monopoly prices.
        """
        p0 = np.ones((1, self.n)) * 3 * self.c
        p_competitive = fsolve(self.foc, p0)
        p_monopoly = fsolve(self.foc_monopoly, p0)
        return p_competitive, p_monopoly

    def init_actions(self):
        """
        Initialize the discrete action space (possible prices).

        Returns:
        -------
        ndarray
            Array of possible prices.
        """
        a = np.linspace(min(self.p_minmax[0]), max(self.p_minmax[1]), self.k - 2)
        delta = a[1] - a[0]
        A = np.linspace(min(a) - delta, max(a) + delta, self.k)
        return A

    def init_state(self):
        """
        Initialize the state space dimensions and initial state.

        Returns:
        -------
        tuple
            A tuple containing state space dimensions and initial state.
        """
        sdim = (self.k, self.k)
        s0 = np.zeros(len(sdim)).astype(int)
        return sdim, s0

    def compute_profits(self, p):
        """
        Compute profits for each firm given their prices.

        Parameters:
        ----------
        p : ndarray
            Array of prices set by each firm.

        Returns:
        -------
        ndarray
            Array of profits for each firm.
        """ 
        d = self.demand(p)
        pi = (p - self.c) * d
        return pi

    def init_PI(game):
        """
        Initialize the profit matrix for all possible states and actions.

        Returns:
        -------
        ndarray
            3D array of profits for all possible states and actions.
        """
        PI = np.zeros(game.sdim + (game.n,))
        for s in product(*[range(i) for i in game.sdim]):
            p = np.asarray(game.A[np.asarray(s)])
            PI[s] = game.compute_profits(p)
        return PI
    
    def show_stats(self):
        """
        Display competitive and monopoly prices.
        """
        p_competitive, p_monopoly = self.compute_p_competitive_monopoly()
        print(f"Competitive Price: {p_competitive}")
        print(f"Monopoly Price: {p_monopoly}")
    
    def check_convergence(self, game, t, stable1, stable2):
        """
        Check if the game has converged.

        Parameters:
        ----------
        t : int
            Current iteration number.
        stable1 : int
            Number of stable periods for algorithm 1.
        stable2 : int
            Number of stable periods for algorithm 2.

        Returns:
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
        if t == game.tmax-1:
            if stable1 > game.tstable:
                print("Algorithm 1 : Converged. Algorithm 2: Not Converged")
                return True
            elif stable2 > game.tstable:
                print("Algorithm 1 : Not Converged. Algorithm 2: Converged")
                return True
            
            print('ERROR! Not Converged!')
            return True
        return False
    
    def simulate_game(self, Agent1, Agent2, game):

        """
        Simulate the game between two agents.

        Parameters:
        ----------
        Agent1 : object
            First agent with pick_strategies and update_function methods.
        Agent2 : object
            Second agent with pick_strategies and update_function methods.

        Returns:
        -------
        tuple
            A tuple containing the final state, all visited states, and all actions taken.
        """
        s = game.s0
        stable1 = 0
        stable2 = 0
        stable_state0 = 0
        stable_state1 = 0
        all_visited_states = []
        all_actions = []
        for t in range(int(game.tmax)):
           # print(f"t = {t} ----------------------------------------------------------------------------------------")
            a1 = Agent1.pick_strategies(game, s, t)[0]
            a2 = Agent2.pick_strategies(game, s[::-1], t)[0]
            a = (a1, a2)
            all_actions.append(a)
            # print(a)
            pi1 = game.PI[a]
            pi2 = game.PI[a[::-1]]
            s1 = a
            same_state0 = (s[0] == s1[0])
            stable_state0 = (stable_state0 + same_state0)*same_state0

            same_state1 = (s[1] == s1[1])
            stable_state1 = (stable_state1 + same_state1)*same_state1

            _, stable1 = Agent1.update_function(game, s, a, s1, pi1, stable1, t)
            _, stable2 = Agent2.update_function(game, s[::-1], a[::-1], s1[::-1], pi2, stable2, t)
            s = s1
            all_visited_states.append(s1)
            if game.check_convergence(game, t, stable1, stable2):
                break
        return game, s, all_visited_states, all_actions

    
