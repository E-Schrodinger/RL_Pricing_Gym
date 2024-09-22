## RL gym

### TODO

 - Add a softmax policy for choosing Actions
 - Fix batch SARSA, completely broken
 - Add Regret from Professor Arnoud's paper.
 - Do more testing on convergence of Batch_SARSA
 
### Updates (22/09/2024)
 - Tried to fix Batch SARSA, realized it was completely broken.
 - Make graphs to obeserve how average prices change for aritficially created deviations.

### Updates (20/09/2024)
- Added docstrings to the code.

### Updates (19/09/2024)
 - Made beta Agent specific
 - Added PM.Print_Q() function to show Q-Values as a reward punishment scheme. (Only works for 2 player environments)
 - Changed the heatmap to show the joint state distribution
 - Added normalized delta from Calvano.
 - Fixed Dec_Q
 - Made Pricing_metrics more simulation efficient (runs only one set of simulations)
 - Fixed Pricing_Metrics to a single environment and Agent pairs per initialization.
 - Added Q_value checks for Agents in Pricing Metrics
 - Removed Herobrine