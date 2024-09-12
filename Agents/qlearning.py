"""
Q-learning Functions
"""

import sys
import numpy as np







# def check_convergence(game, t, stable):
#     """Check if game converged"""
#     if (t % game.tstable == 0) & (t > 0):
#         sys.stdout.write("\rt=%i" % t)
#         sys.stdout.flush()
#     if stable > game.tstable:
#         print('Converged!')
#         return True
#     if t == game.tmax:
#         print('ERROR! Not Converged!')
#         return True
#     return False


# def simulate_game(game):
#     """Simulate game"""
#     s = game.s0
#     stable = 0
#     # Iterate until convergence
#     for t in range(int(game.tmax)):
#         a = pick_strategies(game, s, t)
#         pi = game.PI[tuple(a)]
#         s1 = a
#         game.Q, stable = update_q(game, s, a, s1, pi, stable)
#         s = s1
#         if check_convergence(game, t, stable):
#             print(game.Q)
#             break
#     return game

class Q_Learning:
    def __init__(self, game, **kwargs):

        self.delta = kwargs.get('delta', 0.95)
        self.epsilon = kwargs.get('epsilon', 0.1)

        self.Q = self.init_Q(game)

    def init_Q(self, game):
        """Initialize Q function (n x #s x k)"""
        Q = np.zeros((game.n,) + game.sdim + (game.k,))
        for n in range(game.n):
            pi = np.mean(game.PI[:, :, n], axis=1 - n)
            Q[n] = np.tile(pi, game.sdim + (1,)) / (1 - self.delta)
        return Q
    
    def pick_strategies(self, game, s, t):
        """Pick strategies by exploration vs exploitation"""
        a = np.zeros(game.n).astype(int)
        pr_explore = max(self.epsilon, np.exp(- t * game.beta))
        e = (pr_explore > np.random.rand(game.n))
        for n in range(game.n):
            if e[n]:
                a[n] = np.random.randint(0, game.k)
            else:
                a[n] = np.argmax(self.Q[(n,) + tuple(s)])
        return a
    
    def update_function(self, game, s, a, s1, pi, stable, t):
        """Update Q matrix"""
        for n in range(game.n):
            subj_state = (n,) + tuple(s) + (a[n],)
            old_value = self.Q[subj_state]
            max_q1 = np.max(self.Q[(n,) + tuple(s1)])
            new_value = pi[n] + self.delta * max_q1
            old_argmax = np.argmax(self.Q[(n,) + tuple(s)])
            self.Q[subj_state] = (1 - game.alpha) * old_value + game.alpha * new_value
            # Check stability
            new_argmax = np.argmax(self.Q[(n,) + tuple(s)])
            same_argmax = (old_argmax == new_argmax)
            stable = (stable + same_argmax) * same_argmax
        return self.Q, stable

