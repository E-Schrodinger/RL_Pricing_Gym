import numpy as np
import copy

class SGD:
    """
    Stochastic Gradient Descent (SGD) Agent for Reinforcement Learning in Game Theory.

    This class implements an SGD-based agent that can operate in either discrete or continuous
    action spaces. It updates its policy parameters using stochastic gradient ascent based on
    observed rewards.

    Attributes
    ----------
    gamma : float
        Discount factor for future rewards (default: 0.95).
    alpha : float
        Learning rate for updating policy parameters (default: 0.01).
    tau : float
        Temperature parameter for the softmax function in discrete action spaces (default: 1.0).
    action_space_type : str
        Type of action space: 'discrete' or 'continuous' (default: 'discrete').
    a1_prices : ndarray or None
        Array of possible action prices for discrete action spaces (default: None).
    action_low : float or None
        Minimum action value for continuous action spaces (default: None).
    action_high : float or None
        Maximum action value for continuous action spaces (default: None).
    cal_k : int
        Number of discrete actions to calculate if a1_prices is not provided (default: 15).
    lump_tol : float
        Tolerance for lumping new opponent prices in augmented space (default: 0.05).
    space_type : str
        Type of state space management: 'default' or 'augment' (default: 'default').
    a1_space : ndarray
        Array representing the action space for discrete actions.
    k : int
        Number of possible discrete actions.
    action_dim : int
        Dimension of the action space (number of possible actions for discrete, 1 for continuous).
    price_state_space : ndarray
        Array representing the state space based on action prices.
    state_dim : int
        Dimension of the state space (fixed at 2: own and opponent's last prices).
    policy_params : dict
        Dictionary mapping states to policy parameters (preferences for discrete, shared for continuous).
    s0 : float or tuple
        Initial state of the agent.
    a_price : float or None
        Last chosen action price.
    stable : int
        Counter for consecutive stable updates indicating convergence.
    mean : float
        Mean parameter for the Gaussian policy in continuous action spaces.
    std_dev : float
        Standard deviation parameter for the Gaussian policy in continuous action spaces.
    last_state : tuple
        Last observed state.
    last_action_index : int
        Index of the last chosen action in discrete action spaces.
    last_action_probs : ndarray
        Probability distribution over actions from the last policy in discrete action spaces.
    last_action : float
        Last chosen action in continuous action spaces.
    last_mean : float
        Mean used in the last action selection for continuous action spaces.
    last_std_dev : float
        Standard deviation used in the last action selection for continuous action spaces.
    """

    def __init__(self, game, **kwargs):
        """
        Initialize the SGD agent.

        Parameters
        ----------
        game : object
            The game environment.
        **kwargs : dict
            Additional parameters to override default values.
        """
        # Learning parameters
        self.gamma = kwargs.get('gamma', 0.95)    # Discount factor
        self.alpha = kwargs.get('alpha', 0.01)    # Learning rate
        self.tau = kwargs.get('tau', 1.0)         # Temperature for softmax (used in discrete actions)

        # Action space parameters
        self.action_space_type = kwargs.get('action_space_type', 'discrete')  # 'discrete' or 'continuous'
        self.a1_prices = kwargs.get("a1_prices", None)
        self.action_low = kwargs.get('action_low', None)    # Minimum action value (for continuous actions)
        self.action_high = kwargs.get('action_high', None)  # Maximum action value (for continuous actions)
        self.cal_k = kwargs.get("cal_k", 15)
        self.lump_tol = kwargs.get("lump_tol", 0.05)
        self.space_type = kwargs.get("space_type", "default")  # 'default' or 'augment'

        if self.action_space_type == 'discrete':
            self.a1_space = self.make_action_space(game)
            self.k = self.a1_space.shape[0]
            self.action_dim = self.k  # Number of possible actions
        elif self.action_space_type == 'continuous':
            if self.action_low is None or self.action_high is None:

                # Set default action bounds
                self.action_low = 1
                self.action_high = 2
            # For continuous actions, action_dim is 1
            self.action_dim = 1
        else:
            raise ValueError("action_space_type must be 'discrete' or 'continuous'")

        # Initialize state space
        self.price_state_space = copy.copy(self.a1_space) if self.action_space_type == 'discrete' else np.array([self.action_low, self.action_high])

        # Define state dimensions
        self.state_dim = 2  # State includes own and opponent's last prices

        # Initialize policy parameters
        # For simplicity, we'll use a table mapping states to preferences over actions (discrete)
        # or to distribution parameters (mean and std_dev) for continuous actions
        self.policy_params = {}
        self.initialize_policy_params()

        # Initialize s0 (initial state)
        self.s0 = self.initialize_state()

        # Additional attributes
        self.a_price = None
        self.stable = 0  # Stability counter

    def make_action_space(self, game):
        """
        Create the action space for discrete actions.

        If a1_prices are not provided, compute a range of prices based on the game's competitive and monopoly prices.

        Parameters
        ----------
        game : object
            The game environment.

        Returns
        -------
        ndarray
            Array of action prices.
        """
        if self.a1_prices is None:
            p_competive, p_monopoly = game.compute_p_competitive_monopoly()
            a = np.linspace(min(p_competive), max(p_monopoly), self.cal_k - 2)
            delta = a[1] - a[0] if len(a) > 1 else 0.1
            self.a1_prices = np.linspace(min(a) - delta, max(a) + delta, self.cal_k)
        else:
            self.a1_prices = np.array(self.a1_prices)
        return self.a1_prices

    def initialize_state(self):
        """
        Initialize the starting state of the agent.

        Returns
        -------
        float or tuple
            Initial price for discrete actions or midpoint for continuous actions.
        """
        # Initialize s0 to be the default starting state
        # Start with the first possible price for the agent or the midpoint for continuous actions
        if self.action_space_type == 'discrete':
            initial_price = self.a1_space[0]
        else:
            initial_price = (self.action_high + self.action_low) / 2.0
        return initial_price

    def initialize_policy_params(self):
        """
        Initialize the policy parameters for all possible states.

        For discrete action spaces, initializes a preference vector for each state.
        For continuous action spaces, initializes shared mean and standard deviation.
        """
        # Initialize policy parameters for all possible states
        # For simplicity, initialize preferences to zeros (discrete)
        # or initialize mean and std_dev for continuous actions
        if self.action_space_type == 'discrete':
            for own_price in self.a1_space:
                for opp_price in self.price_state_space:
                    state = (own_price, opp_price)
                    self.policy_params[state] = np.zeros(self.action_dim)  # Preferences over actions
        elif self.action_space_type == 'continuous':
            # In continuous action space, we cannot store policy parameters for all possible states
            # Instead, we use shared policy parameters
            self.mean = (self.action_high + self.action_low) / 2.0  # Initial mean
            self.std_dev = 1.0  # Initial standard deviation
        else:
            raise ValueError("action_space_type must be 'discrete' or 'continuous'")

    def reset(self, game):
        """
        Reset the agent's policy parameters and stability counter.

        Parameters
        ----------
        game : object
            The game environment.
        """
        # Reset policy parameters and stability counter
        if self.action_space_type == 'discrete':
            self.price_state_space = copy.copy(self.a1_space)
            self.initialize_policy_params()
        elif self.action_space_type == 'continuous':
            self.initialize_policy_params()
        # Reinitialize s0 if needed
        self.s0 = self.initialize_state()
        # Reset stability counter
        self.stable = 0

    def get_state_key(self, p):
        """
        Generate a state key based on the current player profiles or actions.

        For discrete action spaces, maps opponent's price to the closest index.
        For continuous action spaces, uses the actual state tuple.

        Parameters
        ----------
        p : tuple
            Current players' profiles or actions.

        Returns
        -------
        tuple
            The state key used to access policy parameters.
        """
        # For discrete action space, map opponent's price to an index in price_state_space
        # For continuous action space, use the actual state tuple
        own_price = p[0]
        opp_price = p[1]

        if self.action_space_type == 'discrete':
            opp_price_index = self.get_index_2(opp_price)
            opp_price = self.price_state_space[opp_price_index]
            state = (own_price, opp_price)
            # Initialize policy parameters for new states if necessary
            if state not in self.policy_params:
                self.policy_params[state] = np.zeros(self.action_dim)
        elif self.action_space_type == 'continuous':
            # In continuous state space, policy parameters are shared
            state = (own_price, opp_price)
        else:
            raise ValueError("action_space_type must be 'discrete' or 'continuous'")

        return state

    def get_index_2(self, p2):
        """
        Get the index of the opponent's price based on the space type.

        Parameters
        ----------
        p2 : float
            Opponent's price.

        Returns
        -------
        int
            Index of the opponent's price in the state space.
        """
        if self.space_type == 'default':
            return self.find_closest_index(p2)
        elif self.space_type == 'augment':
            return self.make_new_index(p2)
        else:
            print(f"ERROR! space_type {self.space_type} not recognized")
            return None

    def find_closest_index(self, p2):
        """
        Find the index of the closest value to p2 in the price state space.

        Parameters
        ----------
        p2 : float
            Opponent's price.

        Returns
        -------
        int
            Index of the closest price.
        """
        # Find the index of the closest value to p2 in price_state_space
        closest_index = np.abs(self.price_state_space - p2).argmin()
        return closest_index

    def make_new_index(self, p2):
        """
        Create a new index for the opponent's price if it is not within lump_tol.

        Parameters
        ----------
        p2 : float
            Opponent's price.

        Returns
        -------
        int
            Index of the opponent's price in the updated state space.
        """
        if any(abs(x - p2) <= self.lump_tol for x in self.price_state_space):
            return self.find_closest_index(p2)
        else:
            # Append new opponent price to price_state_space
            self.price_state_space = np.append(self.price_state_space, p2)
            # Initialize policy parameters for new states
            for own_price in self.a1_space:
                state = (own_price, p2)
                self.policy_params[state] = np.zeros(self.action_dim)
            return len(self.price_state_space) - 1

    def pick_strategies(self, game, p, t):
        """
        Choose actions based on the current policy and exploration strategy.

        For discrete actions, uses a softmax policy.
        For continuous actions, samples from a Gaussian distribution.

        Parameters
        ----------
        game : object
            The game environment.
        p : tuple
            Current players' profiles or actions.
        t : int
            Current time step.

        Returns
        -------
        float
            Chosen action price.
        """
        # Get the current state key
        state = self.get_state_key(p)

        if self.action_space_type == 'discrete':
            # Get preferences for the current state
            preferences = self.policy_params[state]

            # Compute action probabilities using softmax
            action_probs = self.softmax(preferences, self.tau)

            # Choose action based on the policy
            action_index = np.random.choice(self.action_dim, p=action_probs)
            a_price = self.a1_space[action_index]

            # Store action and state for use in update_function
            self.last_state = state
            self.last_action_index = action_index
            self.last_action_probs = action_probs

        elif self.action_space_type == 'continuous':
            # Use shared policy parameters (mean and std_dev)
            mean = self.mean
            std_dev = self.std_dev

            # Sample action from Gaussian distribution
            a_price = np.random.normal(mean, std_dev)
            # Optionally, clip the action to the action bounds
            a_price = np.clip(a_price, self.action_low, self.action_high)

            # Store action and state for use in update_function
            self.last_state = state
            self.last_action = a_price
            self.last_mean = mean
            self.last_std_dev = std_dev

        else:
            raise ValueError("action_space_type must be 'discrete' or 'continuous'")

        self.a_price = a_price
        return a_price

    def softmax(self, preferences, tau):
        """
        Compute softmax probabilities with temperature tau.

        Parameters
        ----------
        preferences : ndarray
            Preference values for each action.
        tau : float
            Temperature parameter to control randomness.

        Returns
        -------
        ndarray
            Probability distribution over actions.
        """
        # Compute softmax probabilities with temperature tau
        prefs = preferences / tau
        max_pref = np.max(prefs)  # For numerical stability
        exp_prefs = np.exp(prefs - max_pref)
        sum_exp_prefs = np.sum(exp_prefs)
        probs = exp_prefs / sum_exp_prefs if sum_exp_prefs > 0 else np.ones_like(preferences) / len(preferences)
        return probs

    def update_function(self, game, p, a_prices, pi, stable, t, tol=1e-2):
        """
        Update the policy parameters based on the observed transition and reward.

        Implements stochastic gradient ascent for policy optimization.

        Parameters
        ----------
        game : object
            The game environment.
        p : tuple
            Previous players' profiles or actions.
        a_prices : tuple
            Chosen action prices for the players.
        pi : float or ndarray
            Observed payoff for the agent.
        stable : int
            Current stability counter.
        t : int
            Current time step.
        tol : float, optional
            Tolerance for considering policy parameters as converged (default: 1e-5).

        Returns
        -------
        tuple
            Updated policy parameters (None for continuous) and the updated stability counter.
        """
        # Get reward
        reward = pi  # Assuming pi is the payoff for this agent

        # Get next state
        next_state = self.get_state_key(a_prices)

        if self.action_space_type == 'discrete':
            # Store old policy parameters for stability check
            old_preferences = self.policy_params[self.last_state].copy()

            # Compute the gradient of the log-policy
            grad_log_policy = -self.last_action_probs
            grad_log_policy[self.last_action_index] += 1

            # Update policy parameters using stochastic gradient ascent
            self.policy_params[self.last_state] += self.alpha * reward * grad_log_policy

            # Check for stability (convergence)
            preference_change = np.linalg.norm(self.policy_params[self.last_state] - old_preferences)
            if preference_change < tol:
                self.stable += 1
            else:
                self.stable = 0

        elif self.action_space_type == 'continuous':
            # Store old policy parameters for stability check
            old_mean = self.mean
            old_std_dev = self.std_dev

            action = self.last_action
            mean = self.mean
            std_dev = self.std_dev

            # Compute gradients
            delta = action - mean
            grad_log_pi_mean = delta / (std_dev ** 2)
            grad_log_pi_std = ((delta ** 2) / (std_dev ** 3)) - (1 / std_dev)

            # Update policy parameters using stochastic gradient ascent
            self.mean += self.alpha * reward * grad_log_pi_mean
            self.std_dev += self.alpha * reward * grad_log_pi_std

            # Ensure std_dev remains positive
            self.std_dev = max(self.std_dev, 1e-3)

            # Check for stability (convergence)
            mean_change = abs(self.mean - old_mean)
            std_change = abs(self.std_dev - old_std_dev)
            total_change = mean_change + std_change
            if total_change < tol:
                self.stable += 1
            else:
                self.stable = 0

        else:
            raise ValueError("action_space_type must be 'discrete' or 'continuous'")

        # Return None and the updated stable counter
        return None, self.stable