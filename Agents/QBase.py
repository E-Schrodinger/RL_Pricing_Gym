
import sys
import numpy as np
import copy
from itertools import product

class QBase:
    def __init__(self, game, **kwargs):

        

        self.a1_prices = kwargs.get("a1_prices", None)
        self.lump_tol = kwargs.get("lump_tol", 0.5)
        self.cal_k = kwargs.get("cal_k", 15)
        self.lump_tol = kwargs.get("lump_tol", 0.05)
        self.Qinit = kwargs.get('Qinit', 'uniform')
        self.delta = kwargs.get('delta', 0.95)
        self.space_type = kwargs.get("space_type", "default")
        self.aug_init = kwargs.get("aug_init", "uniform")


        self.a1_space = self.make_action_space(game)

        self.k = self.a1_space.shape[0]
        self.init_dim = (len(self.a1_space) ,len(self.a1_space))
        self.PI = self.init_PI(game)
        self.s0 = self.a1_space[0]
        self.price_state_space = copy.copy(self.a1_space)
        self.Q = self.make_Q()
        self.Q_val = self.Q.copy()
        self.num = self.make_num()
    

    def make_action_space(self, game):
        if self.a1_prices == None:
            p_competitive, p_monopoly = game.compute_p_competitive_monopoly()
            a = np.linspace(min(p_competitive), max(p_monopoly), self.cal_k - 2)
            delta = a[1] - a[0]
            self.a1_prices = np.linspace(min(a) - delta, max(a) + delta, self.cal_k)
        else:
            self.a1_prices = np.array(self.a1_prices)
        return self.a1_prices
    
    def init_PI(self, game):
        """
        Initialize the profit matrix for all possible states and actions.

        Returns:
        -------
        ndarray
            3D array of profits for all possible states and actions.
        """
        PI = np.zeros( self.init_dim + (2,))
        for s in product(*[range(i) for i in  self.init_dim]):
            p = np.asarray(self.a1_space[np.asarray(s)])
            PI[s] = game.compute_profits(p)
        return PI

    def make_Q(self):
        shape = (len(self.a1_space),) + (len(self.a1_space),) + (len(self.a1_space),)
        if self.Qinit == "uniform":
            Q_init = np.random.rand(*shape)
        elif self.Qinit == "zeros":
            Q_init = np.zeros(shape)
        else:
            Q_init = np.zeros(shape)
             # Calculate mean payoffs across opponent's actions
            pi = np.mean(self.PI[:, :,0], axis=0)
            # Initialize Q-values with discounted mean payoffs
            Q_init = np.tile(pi, self.init_dim + (1,)) / (1 - self.delta)
        return Q_init
    
    def get_index_1(self, p1):
        return np.where(self.a1_space == p1)[0][0]
    
    def get_index_2(self, p2):
        if self.space_type == 'default':
            return self.find_closest_index(p2)
        elif self.space_type == 'augment':
            return self.make_new_index(p2)
        else:
            print(f"ERROR! space_type {self.space_type} not recognized")


    def find_closest_index(self, p2):
        # Find the index of the closest value to p in a1_space
        closest_index = np.abs(self.price_state_space - p2).argmin()
        return closest_index
    
    def make_new_index(self, p2):
        if any(abs(x - p2) <= self.lump_tol for x in self.price_state_space):
            return self.find_closest_index(p2)
        else:
            self.price_state_space = np.append(self.price_state_space, p2)

            self.Q, self.Q_val = self.augment_Q()
            self.num = self.augment_num()

            return self.price_state_space.shape[0]-1
    
    def augment_Q(self):
        shape = (len(self.a1_space),) + (1,) + (len(self.a1_space),)
        
        if self.aug_init == 'zero':
            added_array = np.zeros(shape)
        else:
            added_array = np.random.rand(*shape)
        

        
        return np.hstack((self.Q, added_array)), np.hstack((self.Q_val, added_array))
    
    def make_num(self):
        """
        Initialize the num matrix corresponding to the state space.
        This could represent counts, frequencies, or other numerical metrics.

        Returns:
        -------
        ndarray
            2D array initialized to zeros with the same shape as the state space.
        """
        return np.zeros(self.init_dim, dtype=int)

    def augment_num(self):
        """
        Augment the num matrix to accommodate a new price in the price_state_space.

        This method should be called whenever the price space is augmented.
        """
        old_shape = self.num.shape
        new_size = old_shape[0] + 1  # Assuming square state space

        # Initialize the new num matrix with zeros
        new_num = np.zeros((new_size, new_size), dtype=int)

        # Copy existing data
        new_num[:old_shape[0], :old_shape[1]] = self.num

        # Update self.num
        self.num = new_num
        return self.num

    def update_num(self, state_indices):
        """
        Update the num matrix based on visited state indices.

        Parameters:
        ----------
        state_indices : tuple
            A tuple representing the indices of the current state.
        """
        self.num[state_indices] += 1






    

