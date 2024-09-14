"""
Q-learning Functions
"""

import sys
import numpy as np

    
class Batch_SARSA:
    def __init__(self, game, **kwargs):

        self.delta = kwargs.get('delta', 0.95)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.batch_size = kwargs.get('batch_size', 1000)

        
        self.Q_act = self.init_Q_act(game)
        self.Q_val = self.Q_act.copy()
        self.trans = self.init_trans(game)
        self.num = self.init_num(game)
        self.reward = self.init_reward(game)

        self.X = np.full(self.Q_act.shape, self.epsilon / game.k)
        for n in range(1):
            optimal_actions = np.argmax(self.Q_act[n], axis=-1)
            self.X[n][np.arange(game.sdim[0]), np.arange(game.sdim[1]), optimal_actions] += 1 - self.epsilon


    
    def init_Q_act(self, game):
        """Initialize Q function (n x #s x k)"""
        Q = np.zeros((game.n,) + game.sdim + (game.k,))
        for n in range(game.n):
            pi = np.mean(game.PI[:, :, n], axis=1 - n)
            Q[n] = np.tile(pi, game.sdim + (1,)) / (1 - self.delta)
        return Q
    
    def init_trans(self, game):
        """Initialize transition function (n x #s x #s x k)"""
        Q = np.zeros((game.n,) + game.sdim + game.sdim + (game.k,))
        return Q
    
    def init_num(self, game):
        """Initialize Q function (n x #s x k)"""
        Q = np.zeros((game.n,) + game.sdim + (game.k,))
        return Q
    
    def init_reward(self, game):
        """Initialize Q function (n x #s x k)"""
        Q = np.zeros((game.n,) + game.sdim + (game.k,))
        return Q
    
    def pick_strategies(self, game, s, t):
        """Pick strategies by exploration vs exploitation"""
        a = np.zeros(game.n).astype(int)
        pr_explore = np.exp(- t * game.beta)
        e = (pr_explore > np.random.rand(game.n))
        for n in range(1):
            if e[n]:
                a[n] = np.random.randint(0, game.k)
            else:
                a[n] = np.argmax(self.Q_act[(n,) + tuple(s)])
        return a
    
    def X_function(self, game,s, a):
        for n in range(game.n):
            probabilities = np.zeros(game.n)
            optimal = np.argmax(self.Q_act[(n,) + tuple(s)])
            if a == optimal:
                probabilities[n] = self.epsilon/game.k + 1 - self.epsilon
            else:
                probabilities[n] = self.epsilon/game.k

        return probabilities
    
    def adaption_phase(self, game, s_hat, a_hat, s_prime):
        for n in range(1):
            state = (n,) + tuple(s_hat) + (a_hat[n],)
            
            # Set r_hat
            r_hat = self.reward[state] / max(1, self.num[state])
            
            # Set v_hat
            v_hat = 0
            for b in range(game.k):
                p_b = self.trans[(n,) + tuple(s_prime) + tuple(s_hat) + (a_hat[n],)] / max(1, self.num[state])
                v_hat += p_b * self.Q_val[(n,) + tuple(s_prime) + (b,)]
            
            # Update Q_act
            self.Q_act[state] = (1 - game.alpha) * self.Q_act[state] + game.alpha * (r_hat + self.delta * v_hat)
            
            # Set x(a|s) as epsilon-greedy strategy
            optimal_action = np.argmax(self.Q_act[(n,) + tuple(s_hat)])
            self.X[state] = self.epsilon / game.k
            self.X[(n,) + tuple(s_hat) + (optimal_action,)] += 1 - self.epsilon
            
            # Set Q_val
            self.Q_val[state] = self.Q_act[state]
            
            # Reset counters
            self.trans[(n,) + tuple(s_prime) + tuple(s_hat) + (a_hat[n],)] = 0
            self.num[state] = 0
            self.reward[state] = 0
    
    def update_function(self, game, s, a, s1, pi, stable, t, tol = 1e-1):
        for n in range(1):
            subj_state = (n,) + tuple(s) + (a[n],)
            old_value = self.Q_val[subj_state]
            self.num[subj_state] += 1
            self.trans[(n,) + tuple(s1) + tuple(s) + (a[n],)] += 1

            self.reward[subj_state] += pi[n]
            a_t = 1/(self.num[subj_state]+1)
            Q_merge = 0
            for i in range(game.k):
                Q_merge += self.X_function(game, s,i)*self.Q_val[(n,) + tuple(s1) + (i,)]

            old_argmax = np.argmax(self.Q_val[(n,) + tuple(s)])
            self.Q_val[subj_state] = (1-a_t)*old_value + a_t*(pi[n] + self.delta*Q_merge[n])

            #Check stability
            new_argmax = np.argmax(self.Q_val[(n,) + tuple(s)])
            same_argmax = (old_argmax == new_argmax)
            stable = (stable + same_argmax) * same_argmax

        if (t%self.batch_size == 0):
            for s_hat in np.ndindex(game.sdim):
                for a_hat in np.ndindex((game.k,) * game.n):
                    for s_prime in np.ndindex(game.sdim):
                        self.adaption_phase(game, s_hat, a_hat, s_prime)
            self.trans.fill(0)
            self.num.fill(0)
            self.reward.fill(0)
       
        return self.Q_act, stable


