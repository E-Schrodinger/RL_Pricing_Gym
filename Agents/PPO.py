import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy


class Actor(nn.Module):
    """
    Actor Network for PPO Agent.

    This network takes the current state as input and outputs either
    a probability distribution over possible actions (discrete action space),
    or the parameters (mean and standard deviation) of a Gaussian distribution
    over continuous actions.

    Attributes
    ----------
    fc1 : nn.Linear
        First fully connected layer.
    fc2 : nn.Linear
        Second fully connected layer.
    action_head : nn.Linear
        Output layer that maps to action probabilities or mean.
    log_std : nn.Parameter
        Learnable parameter for the standard deviation (continuous actions).
    softmax : nn.Softmax
        Softmax activation to obtain probabilities (discrete actions).
    """

    def __init__(self, state_dim, action_dim, action_space_type='discrete'):
        """
        Initialize the Actor network.

        Parameters
        ----------
        state_dim : int
            Dimension of the input state.
        action_dim : int
            Number of possible actions (discrete) or dimension of action (continuous).
        action_space_type : str
            Type of action space: 'discrete' or 'continuous'.
        """
        super(Actor, self).__init__()
        self.action_space_type = action_space_type
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        if self.action_space_type == 'discrete':
            self.action_head = nn.Linear(64, action_dim)
            self.softmax = nn.Softmax(dim=-1)
        elif self.action_space_type == 'continuous':
            self.mean_head = nn.Linear(64, action_dim)
            # Learnable parameter for log standard deviation
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            raise ValueError("action_space_type must be 'discrete' or 'continuous'")

    def forward(self, x):
        """
        Forward pass through the Actor network.

        Parameters
        ----------
        x : torch.Tensor
            Input state tensor.

        Returns
        -------
        torch.Tensor
            Action probabilities (discrete) or action mean and std (continuous).
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        if self.action_space_type == 'discrete':
            action_probs = self.softmax(self.action_head(x))
            return action_probs
        elif self.action_space_type == 'continuous':
            mean = self.mean_head(x)
            std = torch.exp(self.log_std)  # Exponentiate log_std to get std
            return mean, std


class Critic(nn.Module):
    """
    Critic Network for PPO Agent.

    This network takes the current state as input and outputs 
    a value estimate of that state.

    Attributes
    ----------
    fc1 : nn.Linear
        First fully connected layer.
    fc2 : nn.Linear
        Second fully connected layer.
    value_head : nn.Linear
        Output layer that maps to a single state value.
    """

    def __init__(self, state_dim):
        """
        Initialize the Critic network.

        Parameters
        ----------
        state_dim : int
            Dimension of the input state.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass through the Critic network.

        Parameters
        ----------
        x : torch.Tensor
            Input state tensor.

        Returns
        -------
        torch.Tensor
            Estimated value of the state.
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        state_value = self.value_head(x)
        return state_value


class PPO_Agent:
    """
    Proximal Policy Optimization (PPO) Agent.

    This agent uses PPO to learn optimal pricing strategies within a game environment.
    It supports both discrete and continuous action spaces.

    Attributes
    ----------
    gamma : float
        Discount factor for rewards.
    epsilon_clip : float
        Clipping parameter for PPO.
    K_epochs : int
        Number of epochs for updating the policy.
    lr_actor : float
        Learning rate for the actor network.
    lr_critic : float
        Learning rate for the critic network.
    buffer_size : int
        Size of the experience buffer.
    batch_size : int
        Batch size for training.
    action_space_type : str
        Type of action space: 'discrete' or 'continuous'.
    a1_prices : np.ndarray or None
        Array of possible prices for agent 1 (discrete actions).
    cal_k : int
        Number of price points in the action space.
    lump_tol : float
        Tolerance level for lumped states.
    space_type : str
        Type of action space to use.
    a1_space : np.ndarray
        Discrete action space (possible prices) for discrete actions.
    action_low : float
        Minimum action value for continuous action spaces.
    action_high : float
        Maximum action value for continuous action spaces.
    k : int
        Number of possible actions (discrete).
    price_state_space : np.ndarray
        Copy of the action space for state representation.
    state_dim : int
        Dimension of the state space.
    action_dim : int
        Dimension of the action space.
    policy_net : Actor
        Actor network for policy.
    value_net : Critic
        Critic network for value estimation.
    optimizer_policy : optim.Optimizer
        Optimizer for the actor network.
    optimizer_value : optim.Optimizer
        Optimizer for the critic network.
    buffer : list
        Experience buffer to store transitions.
    s0 : float
        Initial state.
    a_price : float or None
        Current action price chosen by the agent.
    stable : int
        Counter for stability (convergence) checks.
    """

    def __init__(self, game, **kwargs):
        """
        Initialize the PPO Agent with given or default parameters.

        Parameters
        ----------
        game : object
            The game environment instance.
        **kwargs : dict
            Additional parameters to override default values.
        """
        # PPO parameters
        self.gamma = kwargs.get('gamma', 0.95)
        self.epsilon_clip = kwargs.get('epsilon_clip', 0.2)
        self.K_epochs = kwargs.get('K_epochs', 4)
        self.lr_actor = kwargs.get('lr_actor', 0.0003)
        self.lr_critic = kwargs.get('lr_critic', 0.001)
        self.buffer_size = kwargs.get('buffer_size', 64)
        self.batch_size = kwargs.get('batch_size', 32)

        # Action space parameters
        self.action_space_type = kwargs.get('action_space_type', 'discrete')  # 'discrete' or 'continuous'
        self.a1_prices = kwargs.get("a1_prices", None)
        self.cal_k = kwargs.get("cal_k", 15)
        self.lump_tol = kwargs.get("lump_tol", 0.05)
        self.space_type = kwargs.get("space_type", "default")

        if self.action_space_type == 'discrete':
            self.a1_space = self.make_action_space(game)
            self.k = self.a1_space.shape[0]
            self.price_state_space = copy.copy(self.a1_space)

            # Define state and action dimensions
            self.state_dim = 2  # State includes own and opponent's last prices
            self.action_dim = self.k  # Number of possible actions

        elif self.action_space_type == 'continuous':
            self.action_low = kwargs.get('action_low', None)
            self.action_high = kwargs.get('action_high', None)
            if self.action_low is None or self.action_high is None:
                # Set default action bounds
                self.action_low = 0.0
                self.action_high = 1.0
            # For continuous actions, action_dim is 1
            self.action_dim = 1
            # Initialize action space
            self.price_state_space = np.array([self.action_low, self.action_high])
            # Define state dimensions
            self.state_dim = 2  # State includes own and opponent's last prices
        else:
            raise ValueError("action_space_type must be 'discrete' or 'continuous'")

        # Initialize policy and value networks
        self.policy_net = Actor(self.state_dim, self.action_dim, self.action_space_type)
        self.value_net = Critic(self.state_dim)

        # Initialize optimizers
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=self.lr_actor)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=self.lr_critic)

        # Buffer to store experiences
        self.buffer = []

        # Initialize s0 (initial state)
        self.s0 = self.initialize_state()

        # Additional attributes to match the expected agent interface
        self.a_price = None

        # Initialize stability counter
        self.stable = 0

    def make_action_space(self, game):
        """
        Create the discrete action space based on competitive and monopoly prices.

        Parameters
        ----------
        game : object
            The game environment instance.

        Returns
        -------
        np.ndarray
            Array of possible price actions.
        """
        if self.a1_prices is None:
            p_competitive, p_monopoly = game.compute_p_competitive_monopoly()
            a = np.linspace(min(p_competitive), max(p_monopoly), self.cal_k - 2)
            delta = a[1] - a[0] if len(a) > 1 else 0.1
            self.a1_prices = np.linspace(min(a) - delta, max(a) + delta, self.cal_k)
        else:
            self.a1_prices = np.array(self.a1_prices)
        return self.a1_prices

    def initialize_state(self):
        """
        Initialize the starting state of the agent.

        For PPO, the initial state can consist of the first possible price for both agents.

        Returns
        -------
        float
            Initial price.
        """
        if self.action_space_type == 'discrete':
            initial_price = self.a1_space[0]
        elif self.action_space_type == 'continuous':
            initial_price = (self.action_high + self.action_low) / 2.0
        else:
            raise ValueError("action_space_type must be 'discrete' or 'continuous'")
        return initial_price

    def reset(self, game):
        """
        Reset the agent's experience buffer and state.

        Parameters
        ----------
        game : object
            The game environment instance.
        """
        # Reset buffer and any other variables
        self.buffer = []
        # Reinitialize s0 if needed
        self.s0 = self.initialize_state()
        # Reset stability counter
        self.stable = 0

    def pick_strategies(self, game, p, t):
        """
        Select an action (price) based on the current state using the policy network.

        Parameters
        ----------
        game : object
            The game environment instance.
        p : tuple
            Tuple containing the last prices of both agents.
        t : int
            Current time step.

        Returns
        -------
        float
            Selected price action.
        """
        # State includes own last price and opponent's last price
        s = np.array([p[0], p[1]], dtype=np.float32)
        s_tensor = torch.tensor(s, dtype=torch.float32)

        if self.action_space_type == 'discrete':
            # Get action probabilities from policy network
            with torch.no_grad():
                action_probs = self.policy_net(s_tensor)
            m = torch.distributions.Categorical(action_probs)
            a = m.sample()
            a_price = self.a1_space[a.item()]
            log_prob = m.log_prob(a)
            # Store data in buffer
            self.buffer.append({
                'state': s,
                'action': a.item(),
                'log_prob': log_prob.item(),
                'reward': None,          # To be filled in update_function
                'next_state': None       # To be filled in update_function
            })

        elif self.action_space_type == 'continuous':
            # Get mean and std from policy network
            with torch.no_grad():
                mean, std = self.policy_net(s_tensor)
            mean = mean.squeeze()
            std = std.squeeze()
            # Create normal distribution
            m = torch.distributions.Normal(mean, std)
            # Sample action
            a = m.sample()
            # Apply tanh to bound the actions between -1 and 1
            a_tanh = torch.tanh(a)
            # Scale to action_low and action_high
            a_price = a_tanh.item() * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2
            # Compute log probability (adjusted for Tanh transformation)
            log_prob = m.log_prob(a) - torch.log(1 - a_tanh.pow(2) + 1e-6)
            log_prob = log_prob.sum()

            # Store data in buffer
            self.buffer.append({
                'state': s,
                'action': a.item(),
                'log_prob': log_prob.item(),
                'reward': None,          # To be filled in update_function
                'next_state': None       # To be filled in update_function
            })
        else:
            raise ValueError("action_space_type must be 'discrete' or 'continuous'")

        self.a_price = a_price
        return a_price

    def update_function(self, game, p, a_prices, pi, stable, t, tol=1e-5):
        """
        Update the agent's policy and value networks based on the received reward.

        Parameters
        ----------
        game : object
            The game environment instance.
        p : tuple
            Current prices of both agents.
        a_prices : tuple
            New prices after actions are taken.
        pi : float
            Profit obtained from the current action.
        stable : int
            Current stability counter.
        t : int
            Current time step.
        tol : float, optional
            Tolerance for stability check (default is 1e-5).

        Returns
        -------
        tuple
            - None
            - Updated stability counter.
        """
        # Get reward
        reward = pi  # Assuming pi is the payoff for this agent

        # Get next state (own new price, opponent's new price)
        s_next = np.array([a_prices[0], a_prices[1]], dtype=np.float32)

        # Update the last entry in the buffer with reward and next state
        self.buffer[-1]['reward'] = reward
        self.buffer[-1]['next_state'] = s_next

        # Check if buffer is full
        if len(self.buffer) >= self.buffer_size:
            # Perform PPO update
            self.update_policy(tol)
            # Clear buffer
            self.buffer = []

        # Return None and the updated stable counter
        return None, self.stable

    def update_policy(self, tol):
        """
        Update the policy and value networks using the experiences in the buffer.

        Parameters
        ----------
        tol : float
            Tolerance for checking parameter convergence.
        """
        # Store old policy parameters for stability check
        old_params = [param.clone() for param in self.policy_net.parameters()]

        # Convert buffer to tensors
        states = torch.from_numpy(np.array([item['state'] for item in self.buffer], dtype=np.float32))
        actions = torch.tensor([item['action'] for item in self.buffer], dtype=torch.float32 if self.action_space_type == 'continuous' else torch.long)
        rewards = [item['reward'] for item in self.buffer]
        next_states = torch.from_numpy(np.array([item['next_state'] for item in self.buffer], dtype=np.float32))
        old_log_probs = torch.tensor([item['log_prob'] for item in self.buffer], dtype=torch.float32)

        # Compute discounted rewards (returns)
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Compute advantages
        values = self.value_net(states).squeeze()
        advantages = returns - values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            if self.action_space_type == 'discrete':
                # Get action probabilities
                action_probs = self.policy_net(states)
                m = torch.distributions.Categorical(action_probs)
                log_probs = m.log_prob(actions)

                # Compute ratio
                ratios = torch.exp(log_probs - old_log_probs)

                # Compute surrogate loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages

                # Compute actor loss
                actor_loss = -torch.min(surr1, surr2).mean()

            elif self.action_space_type == 'continuous':
                # Get mean and std from policy network
                mean, std = self.policy_net(states)
                # Create normal distribution
                m = torch.distributions.Normal(mean, std)
                # Compute log probabilities
                log_probs = m.log_prob(actions.unsqueeze(-1))
                log_probs = log_probs.sum(dim=-1)
                # Adjust log probabilities for Tanh squashing
                action_tanh = torch.tanh(actions)
                log_probs -= torch.log(1 - action_tanh.pow(2) + 1e-6)
                log_probs = log_probs.sum(dim=-1)

                # Compute ratio
                ratios = torch.exp(log_probs - old_log_probs)

                # Compute surrogate loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages

                # Compute actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
            else:
                raise ValueError("action_space_type must be 'discrete' or 'continuous'")

            # Compute critic loss
            values = self.value_net(states).squeeze()
            critic_loss = nn.MSELoss()(values, returns)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss

            # Update networks
            self.optimizer_policy.zero_grad()
            self.optimizer_value.zero_grad()
            loss.backward()
            self.optimizer_policy.step()
            self.optimizer_value.step()

        # Check for stability (convergence)
        same_params = True
        for old_param, new_param in zip(old_params, self.policy_net.parameters()):
            if not torch.allclose(old_param, new_param, atol=tol, rtol=tol):
                same_params = False
                break

        if same_params:
            self.stable += 1
        else:
            self.stable = 0
