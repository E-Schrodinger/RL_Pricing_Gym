import sys
import numpy as np
import copy
from itertools import product


class QBase:
    """
    Base class for Q-Learning agents in a pricing game environment.

    This class initializes the action and state spaces, profit matrix, Q-values,
    and provides methods to manage and update the Q-values based on game dynamics.

    Attributes
    ----------
    a1_prices : np.ndarray or None
        Array of possible prices for agent 1.
    lump_tol : float
        Tolerance level for lumping similar states.
    cal_k : int
        Number of discrete price points in the action space.
    Qinit : str
        Initialization method for Q-values ('uniform', 'zeros', or other).
    delta : float
        Discount factor for future rewards.
    space_type : str
        Type of state space handling ('default' or 'augment').
    aug_init : str
        Initialization method for augmented Q-values ('zero' or other).
    a1_space : np.ndarray
        Discrete action space (possible prices).
    k : int
        Number of possible actions.
    init_dim : tuple
        Dimensions of the initial state-action space.
    PI : np.ndarray
        Profit matrix for all possible states and actions.
    s0 : float
        Initial state value.
    state_space : np.ndarray
        Current state space, potentially augmented.
    Q : np.ndarray
        Q-value matrix.
    Q_val : np.ndarray
        Copy of the Q-value matrix for value estimates.
    num : np.ndarray
        Matrix tracking the number of visits to each state.
    """

    def __init__(self, game, **kwargs):
        """
        Initialize the QBase model with given or default parameters.

        Parameters
        ----------
        game : IRP
            The game environment instance.
        **kwargs : dict
            Additional parameters to override default values.
        """
        self.a1_prices = kwargs.get("a1_prices", None)
        self.lump_tol = kwargs.get("lump_tol", 0.5)
        self.cal_k = kwargs.get("cal_k", 15)
        self.lump_tol = kwargs.get("lump_tol", 0.05)  # Note: Overwrites previous lump_tol
        self.Qinit = kwargs.get('Qinit', 'uniform')
        self.delta = kwargs.get('delta', 0.95)
        self.space_type = kwargs.get("space_type", "default")
        self.aug_init = kwargs.get("aug_init", "uniform")

        self.a1_space = self.make_action_space(game)
        self.k = self.a1_space.shape[0]
        self.init_dim = (len(self.a1_space), len(self.a1_space))
        self.PI = self.init_PI(game)
        self.s0 = self.a1_space[0]
        self.state_space = copy.copy(self.a1_space)
        self.Q = self.make_Q()
        self.Q_val = self.Q.copy()
        self.num = self.make_num()
        self.t = 0

    def make_action_space(self, game):
        """
        Create the discrete action space based on competitive and monopoly prices.

        Parameters
        ----------
        game : IRP
            The game environment instance.

        Returns
        -------
        np.ndarray
            Array of possible price actions.
        """
        if self.a1_prices is None:
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

        Parameters
        ----------
        game : IRP
            The game environment instance.

        Returns
        -------
        np.ndarray
            3D array of profits for all possible states and actions.
            Dimensions correspond to (agent1_action, agent2_action, agent).
        """
        PI = np.zeros(self.init_dim + (2,))
        for s in product(*[range(i) for i in self.init_dim]):
            p = np.asarray(self.a1_space[np.asarray(s)])
            PI[s] = game.compute_profits(p)
        return PI

    def make_Q(self):
        """
        Initialize the Q-value matrix based on the specified initialization method.

        Returns
        -------
        np.ndarray
            Initialized Q-value matrix with shape (action_space, action_space, action_space).
        """
        shape = (len(self.a1_space), len(self.a1_space), len(self.a1_space))
        if self.Qinit == "uniform":
            Q_init = np.random.rand(*shape)
        elif self.Qinit == "zeros":
            Q_init = np.zeros(shape)
        else:
            Q_init = np.zeros(shape)
            # Calculate mean payoffs across opponent's actions
            pi = np.mean(self.PI[:, :, 0], axis=0)
            # Initialize Q-values with discounted mean payoffs
            Q_init = np.tile(pi, self.init_dim + (1,)) / (1 - self.delta)
        return Q_init

    def get_index_1(self, p1):
        """
        Get the index of the first agent's price in the action space.

        Parameters
        ----------
        p1 : float
            Price of agent 1.

        Returns
        -------
        int
            Index of the price in the action space.
        """
        return np.where(self.a1_space == p1)[0][0]

    def get_index_2(self, p2):
        """
        Get the index of the second agent's price in the state space based on space_type.

        Parameters
        ----------
        p2 : float
            Price of agent 2.

        Returns
        -------
        int
            Index of the price in the state space.

        Raises
        ------
        ValueError
            If space_type is not recognized.
        # """
        # if self.t == 0:
        #     print(self.space_type)
        if self.space_type == 'default':
            return self.find_closest_index(p2)
        elif self.space_type == 'augment':
            return self.make_new_index(p2)
        else:
            print(f"ERROR! space_type {self.space_type} not recognized")

    def find_closest_index(self, p2):
        """
        Find the index of the closest price to p2 in the state space.

        Parameters
        ----------
        p2 : float
            Target price to find.

        Returns
        -------
        int
            Index of the closest price in the state space.
        """
        closest_index = np.abs(self.state_space - p2).argmin()
        return closest_index

    def make_new_index(self, p2):
        """
        Create a new index for a price if it is not within lump_tol of existing prices.

        Parameters
        ----------
        p2 : float
            New price to be added.

        Returns
        -------
        int
            Index of the new or existing price in the state space.
        # """
        # if self.t == 0:
        #     print(any(abs(x - p2) <= self.lump_tol for x in self.state_space))
        if any(abs(x - p2) <= self.lump_tol for x in self.state_space):
            
            return self.find_closest_index(p2)
        else:
            self.state_space = np.append(self.state_space, p2)
            # if self.t == 0:
            #     print(self.state_space.shape)
            self.Q, self.Q_val = self.augment_Q()
            self.num = self.augment_num()
            return self.state_space.shape[0] - 1

    def augment_Q(self):
        """
        Augment the Q-value matrices to accommodate a new action.

        Returns
        -------
        tuple of np.ndarray
            Augmented Q and Q_val matrices.
        """
        shape = (len(self.a1_space), 1, len(self.a1_space))
        
        if self.aug_init == 'zero':
            added_array = np.zeros(shape)
        else:
            added_array = np.random.rand(*shape)

        return np.hstack((self.Q, added_array)), np.hstack((self.Q_val, added_array))

    def make_num(self):
        """
        Initialize the visit count matrix corresponding to the state space.

        This matrix tracks the number of times each state has been visited.

        Returns
        -------
        np.ndarray
            2D array initialized to zeros with the same shape as the state space.
        """
        return np.zeros(self.init_dim, dtype=int)

    def augment_num(self):
        """
        Augment the visit count matrix to accommodate a new price in the state space.

        This method should be called whenever the price space is augmented.

        Returns
        -------
        np.ndarray
            Updated visit count matrix.
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
        Update the visit count matrix based on visited state indices.

        Parameters
        ----------
        state_indices : tuple
            A tuple representing the indices of the current state.
        """
        self.num[state_indices] += 1