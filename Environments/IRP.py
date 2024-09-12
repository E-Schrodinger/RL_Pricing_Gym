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
    beta : float
        exploration parameter
    delta : float
        discount factor
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
    stable: int
        periods of game stability
    """

    def __init__(self, dem_function = 'default',**kwargs):
        """Initialize game with default values"""
        # Default properties
        self.n = kwargs.get('n', 2)
        self.alpha = kwargs.get('alpha', 0.15)
        self.beta = kwargs.get('beta', 4e-6)
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

        # self.Q_val = self.init_Q_val()
        # self.Q_act = self.init_Q_act()
        # self.trans = self.init_trans()
        # self.num = self.init_num()
        # self.reward = self.init_reward()

        

    def demand(self, p):
        """Computes demand"""
        e = np.exp((self.a - p) / self.mu)
        d = e / (np.sum(e) + np.exp(self.a0 / self.mu))
        return d

    def foc(self, p):
        """Compute first order condition"""
        if self.dem_function == 'default':
            d = self.demand(p)
        else:
            d = self.dem_function(p)
        zero = 1 - (p - self.c) * (1 - d) / self.mu
        return np.squeeze(zero)

    def foc_monopoly(self, p):
        """Compute first order condition of a monopolist"""
        if self.dem_function == 'default':
            d = self.demand(p)
        else:
            d = self.dem_function(p)
        d1 = np.flip(d)
        p1 = np.flip(p)
        zero = 1 - (p - self.c) * (1 - d) / self.mu + (p1 - self.c) * d1 / self.mu
        return np.squeeze(zero)

    def compute_p_competitive_monopoly(self):
        """Computes competitive and monopoly prices"""
        p0 = np.ones((1, self.n)) * 3 * self.c
        p_competitive = fsolve(self.foc, p0)
        p_monopoly = fsolve(self.foc_monopoly, p0)
        return p_competitive, p_monopoly

    def init_actions(self):
        """Get action space of the firms"""
        a = np.linspace(min(self.p_minmax[0]), max(self.p_minmax[1]), self.k - 2)
        delta = a[1] - a[0]
        A = np.linspace(min(a) - delta, max(a) + delta, self.k)
        return A

    def init_state(self):
        """Get state dimension and initial state"""
        sdim = (self.k, self.k)
        s0 = np.zeros(len(sdim)).astype(int)
        return sdim, s0

    def compute_profits(self, p):
        """Compute payoffs"""
        d = self.demand(p)
        pi = (p - self.c) * d
        return pi

    def init_PI(game):
        """Initialize Profits (k^n x kp x n)"""
        PI = np.zeros(game.sdim + (game.n,))
        for s in product(*[range(i) for i in game.sdim]):
            p = np.asarray(game.A[np.asarray(s)])
            PI[s] = game.compute_profits(p)
        return PI
    
    def check_convergence(self, game, t, stable1, stable2):
        """Check if game converged"""
        if (t % game.tstable == 0) & (t > 0):
            sys.stdout.write("\rt=%i" % t)
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
        s = game.s0
        stable1 = 0
        stable2 = 0
        for t in range(int(game.tmax)):
            a1 = Agent1.pick_strategies(game, s, t)[0]
            a2 = Agent2.pick_strategies(game, s[::-1], t)[0]
            a = (a1, a2)
            pi1 = game.PI[a]
            pi2 = game.PI[a[::-1]]
            s1 = a
            _, stable1 = Agent1.update_function(game, s, a, s1, pi1, stable1, t)
            _, stable2 = Agent2.update_function(game, s, a[::-1], s1[::-1], pi2, stable2, t)
            s = s1
            if game.check_convergence(game, t, stable1, stable2):
                print(f"Q-values difference: {np.max(np.abs(Agent1.Q - Agent2.Q[::-1, ::-1]))}")
                break
        return game



    # def init_Q(game):
    #     """Initialize Q function (n x #s x k)"""
    #     Q = np.zeros((game.n,) + game.sdim + (game.k,))
    #     for n in range(game.n):
    #         pi = np.mean(game.PI[:, :, n], axis=1 - n)
    #         Q[n] = np.tile(pi, game.sdim + (1,)) / (1 - game.delta)
    #     return Q
    
    # def init_Q_val(game):
    #     """Initialize Q function (n x #s x k)"""
    #     Q = np.zeros((game.n,) + game.sdim + (game.k,))
    #     return Q
    
    # def init_Q_act(game):
    #     """Initialize Q function (n x #s x k)"""
    #     Q = np.zeros((game.n,) + game.sdim + (game.k,))
    #     return Q
    
    # def init_trans(game):
    #     """Initialize transition function (n x #s x #s x k)"""
    #     Q = np.zeros((game.n,) + game.sdim + game.sdim + (game.k,))
    #     return Q
    
    # def init_num(game):
    #     """Initialize Q function (n x #s x k)"""
    #     Q = np.zeros((game.n,) + game.sdim + (game.k,))
    #     return Q
    
    # def init_reward(game):
    #     """Initialize Q function (n x #s x k)"""
    #     Q = np.zeros((game.n,) + game.sdim + (game.k,))
    #     return Q
    
    
