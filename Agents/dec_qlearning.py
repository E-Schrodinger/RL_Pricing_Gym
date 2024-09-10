"""
Q-learning Functions
"""

import sys
import numpy as np


def pick_strategies(game, s, t):
    """Pick strategies by exploration vs exploitation"""
    a = np.zeros(game.n).astype(int)
    pr_explore = game.epsilon
    e = (pr_explore > np.random.rand(game.n))
    for n in range(game.n):
        if e[n]:
            a[n] = np.random.randint(0, game.k)
        else:
            a[n] = np.argmax(game.Q_act[(n,) + tuple(s)])
    return a


def X(game,s, a):
    for n in range(game.n):
        probabilities = np.zeros(game.n)
        optimal = np.argmax(game.Q_act[(n,) + tuple(s)])
        if a == optimal:
            probabilities[n] = game.epsilon/game.k + 1 - game.epsilon
        else:
            probabilities[n] = game.epsilon/game.k

    return probabilities




def update_q(game, s, a, s1, pi, stable):
    for n in range(game.n):
        subj_state = (n,) + tuple(s) + (a[n],)
        old_value = game.Q_val[subj_state]
        game.num[subj_state] += 1
        game.trans[(n,) + tuple(s1) + tuple(s) + (a[n],)] += 1

        game.reward[subj_state] += pi[n]
        a_t = 1/(game.num[subj_state]+1)
        Q_merge = 0
        for i in range(game.k):
            Q_merge += X(game, s,i)*game.Q_val[(n,) + tuple(s1) + (i,)]

        old_argmax = np.argmax(game.Q_val[(n,) + tuple(s)])
        game.Q_val[subj_state] = (1-a_t)*old_value + a_t*(pi[n] + game.delta*Q_merge[n])

        #Check stability
        new_argmax = np.argmax(game.Q_val[(n,) + tuple(s)])
        same_argmax = (old_argmax == new_argmax)
        stable = (stable + same_argmax) * same_argmax
    return game.Q_val, stable





def adaption_phase(game, s_hat, a_hat, s_prime):
    for n in range(game.n):
        state = (n,) + tuple(s_hat) + (a_hat[n],)
        
        # Set r_hat
        r_hat = game.reward[state] / max(1, game.num[state])
        
        # Set v_hat
        v_hat = 0
        for b in range(game.k):
            p_b = game.trans[(n,) + tuple(s_prime) + tuple(s_hat) + (a_hat[n],)] / max(1, game.num[state])
            v_hat += p_b * game.Q_val[(n,) + tuple(s_prime) + (b,)]
        
        # Update Q_act
        game.Q_act[state] = (1 - game.alpha) * game.Q_act[state] + game.alpha * (r_hat + game.delta * v_hat)
        
        # Set x(a|s) as epsilon-greedy strategy
        optimal_action = np.argmax(game.Q_act[(n,) + tuple(s_hat)])
        game.X[state] = game.epsilon / game.k
        game.X[(n,) + tuple(s_hat) + (optimal_action,)] += 1 - game.epsilon
        
        # Set Q_val
        game.Q_val[state] = game.Q_act[state]
        
        # Reset counters
        game.trans[(n,) + tuple(s_prime) + tuple(s_hat) + (a_hat[n],)] = 0
        game.num[state] = 0
        game.reward[state] = 0

def simulate_game(game, batch_size=1000):
    """Simulate game using Sample-Batch Temporal-Difference Learning"""
    s = game.s0
    stable_count = 0
    
    # Initialize q_act(s, a) = q_val(s, a) randomly
    game.Q_act = np.random.rand(*game.Q_act.shape)
    game.Q_val = game.Q_act.copy()
    
    # Initialize p(s'|a, s), n(s, a), and r(s, a) to zero
    game.trans.fill(0)
    game.num.fill(0)
    game.reward.fill(0)
    
    # Set x(a|s) as epsilon-greedy strategy from q_act(s, a)
    game.X = np.full(game.Q_act.shape, game.epsilon / game.k)
    for n in range(game.n):
        optimal_actions = np.argmax(game.Q_act[n], axis=-1)
        game.X[n][np.arange(game.sdim[0]), np.arange(game.sdim[1]), optimal_actions] += 1 - game.epsilon
    
    # Store previous Q_act for comparison
    prev_Q_act = game.Q_act.copy()
    
    # Iterate until convergence or maximum iterations
    for t in range(int(game.tmax)):
        print(t)

        # Interaction phase
        for _ in range(batch_size):
            a = pick_strategies(game, s, t)
            pi = game.PI[tuple(a)]
            s1 = a
            game.Q_val, _ = update_q(game, s, a, s1, pi, 0)  # We don't use the stable return value here
            s = s1
        
        # Adaption phase
        for s_hat in np.ndindex(game.sdim):
            for a_hat in np.ndindex((game.k,) * game.n):
                for s_prime in np.ndindex(game.sdim):
                    adaption_phase(game, s_hat, a_hat, s_prime)
        
        # Check for stability
        if np.allclose(game.Q_act, prev_Q_act, rtol=1e-5, atol=1e-8):
            stable_count += 1
        else:
            stable_count = 0
        
        # Update prev_Q_act for next iteration
        prev_Q_act = game.Q_act.copy()
        
        # Check for convergence
        if stable_count >= game.tstable:
            print(f'Converged after {t+1} iterations!')
            break
    
    if t == game.tmax - 1:
        print('ERROR! Not Converged!')
    
    return game
    
    



