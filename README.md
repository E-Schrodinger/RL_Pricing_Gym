# RL Pricing Gym

A reinforcement learning gym environment designed to study algorithmic pricing and potential collusive behavior in duopoly markets. This framework enables systematic investigation of how different pricing algorithms interact in various market settings.

## Authors
- Vedant Mainkar
- Dr. Janusz Meylahn (Supervisor)
- Ben Meylahn (Supervisor)

## Features

The gym provides a flexible architecture with Environment (market conditions, reward functions), Agent (pricing algorithms), and Metrics (behavior analysis) components. It supports interactions between discrete and continuous space algorithms through novel state space management.

### Implemented Agents
Q-Learning, Decentralized Q-Learning, EXP3, Stochastic Gradient Descent (SGD), and Proximal Policy Optimization (PPO).

### Analysis Tools
Average prices and profits, state heat maps, reward-punishment scheme analysis, regret calculation, and adjacency matrix visualization.

## Usage Example

Here's an example of how to set up and run a simulation comparing Q-Learning against EXP3:

### Environment Setup
```python
# Initialize the Logit demand model environment
# tmax: Maximum steps per learning trajectory (1 million)
# tstable: Steps required for stability check (100,000)
game = IRP(tmax=1000000, tstable=100000)
```

### Agent Configuration
```python
# Define discrete action space (possible prices)
# Range from 1.0 to 2.0 with 0.125 increments
d_action_space = [1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2]

# Initialize Q-Learning agent
Agent1 = Q_Learning(game, 
                    beta=0.0001,  # Exploration decay rate
                    Qinit='uniform',  # Initialize Q-values uniformly
                    a1_prices=d_action_space)  # Available prices

# Initialize EXP3 agent
Agent2 = Exp3(game,
              beta=0.0001,  # Learning rate
              Qinit='uniform',  # Initialize weights uniformly
              a1_prices=d_action_space)  # Available prices
```

### Running Simulations
```python
# Initialize simulation
# iterations: Number of learning trajectories (48)
# ts: Time interval for recording data (every 1000 steps)
SM = Simulations(game, Agent1, Agent2, iterations=48, ts=1000)

# Get results
# a1_list: List of Agent1's actions across all trajectories
# a2_list: List of Agent2's actions across all trajectories
a1_list, a2_list = SM.get_values()
```

### Analysis
```python
# Calculate average prices and profits over stable period
# ts: Time interval used in data recording
stats = average_price_and_profit(game, a1_list, a2_list, ts=1000)

# Calculate regret for both agents
regret = find_regret(game, Agent1, Agent2, a1_list, a2_list)

# Generate heat map of joint actions
# time_step: Specific time step to analyze
state_heatmap(game, Agent1, Agent2, a1_list, a2_list, time_step)
```

## Parameters

- `tmax`: Maximum number of steps per learning trajectory
- `tstable`: Number of steps required for stability
- `beta`: Learning rate/exploration decay rate depending on the algorithm
- `Qinit`: Q-table/weights initialization method ('uniform', 'zero', or 'mean')
- `ts`: Time step interval for recording data
- `iterations`: Number of learning trajectories to run
- `time_step`: Specific time step for analysis in metrics