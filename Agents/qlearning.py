"""
Q-learning Functions
"""

import sys
import numpy as np
import copy


class Q_Learning:
    def __init__(self, game, **kwargs):

        self.delta = kwargs.get('delta', 0.95)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.beta = kwargs.get('beta', 4e-6)

        self.Q = self.init_Q(game)

    def init_Q(self, game):
        """Initialize Q function (n x #s x k)"""
        Q = np.zeros((game.n,) + game.sdim + (game.k,))
        for n in range(game.n):
            pi = np.mean(game.PI[:, :, n], axis=1 - n)
            Q[n] = np.tile(pi, game.sdim + (1,)) / (1 - self.delta)
        return Q
    
    def reset(self, game):
        self.Q = self.init_Q(game)
    
    def pick_strategies(self, game, s, t):
        """Pick strategies by exploration vs exploitation"""
        a = np.zeros(game.n).astype(int)
        pr_explore = np.exp(- t * self.beta)
        # pr_explore = 0.1
        e = (pr_explore > np.random.rand(game.n))
        for n in range(1):
            if e[n]:
                a[n] = np.random.randint(0, game.k)
            else:
                a[n] = np.argmax(self.Q[(n,) + tuple(s)])
        return a
    
    def update_function(self, game, s, a, s1, pi, stable, t, tol = 1e-1):
        """Update Q matrix"""
        for n in range(1):
            subj_state = (n,) + tuple(s) + (a[n],)
            old_q = self.Q[0].copy()
            old_value = self.Q[subj_state]
            max_q1 = np.max(self.Q[(n,) + tuple(s1)])
            new_value = pi[n] + self.delta * max_q1
            self.Q[subj_state] = (1 - game.alpha) * old_value + game.alpha * new_value
            # Check stability
            same_q = np.allclose(old_q, self.Q[0], tol)
            stable = (stable + same_q) * same_q
        return self.Q, stable

