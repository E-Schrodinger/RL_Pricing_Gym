## RL gym

### TODO
 - Check on Q-learning convergence rate
 - Fix my graphics
 - Add state space to Agent class
 - Add time dependent metrics
 - Add a softmax policy for choosing Actions
 - Fix batch SARSA, completely broken
 - Add Regret from Professor Arnoud's paper.
 - Do more testing on convergence of Batch_SARSA

### Updates (26/09/2024)
 - Rewrote some of the code to remove redundancies.
 - Added Regret metric.

### Updates (23/09/2024)
 - Bug fixes.
 - Added different initializations for Q-learning. (uniform, calvano, zero)
 - Implemented Decentralized Q-learning
 - Added Network Graph which displays Reward-punishment schemes for states. (Works fine but not visually readable.)

### Updates (22/09/2024)
 - Tried to fix Batch SARSA, realized it was completely broken.
 - Added new Price_Deivation Metric. Will force a deviation in actions and create graph of behavior.

### Updates (20/09/2024)
- Added docstrings to the code.

### Updates (19/09/2024)
 - Made beta Agent specific
 - Added PM.Q_table() function to show Q-Values as a reward punishment scheme. (Only works for 2 player environments)
 - Changed the heatmap to show the joint state distribution
 - Added normalized delta from Calvano.
 - Fixed Dec_Q
 - Made Pricing_metrics more simulation efficient (runs only one set of simulations)
 - Fixed Pricing_Metrics to a single environment and Agent pairs per initialization.
 - Added Q_value checks for Agents in Pricing Metrics
 - Removed Herobrine