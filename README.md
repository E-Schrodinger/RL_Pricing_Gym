## RL gym

### TODO
 - Add good comments (make the code readable)
 - Add a softmax policy for choosing Actions
 - Add Regret from Professor Arnoud's paper.
 - Do more testing on convergence of Batch_SARSA
 - Make graphs to obeserve how average prices change for aritficially created deviations. (Will require a lot of planning before implementation)

### Updates (09/19/2024)
 - Made beta Agent specific
 - Added PM.Print_Q() function to show Q-Values as a reward punishment scheme. (Only works for 2 player environments)
 - Changed the heatmap to show the joint state distribution
 - Added normalized delta from Calvano.
 - Fixed Dec_Q
 - Made Pricing_metrics more simulation efficient (runs only one set of simulations)
 - Fixed Pricing_Metrics to a single environment and Agent pairs per initialization. This was a BIG MISTAKE I just noticed.
 - Added Q_value checks for Agents in Pricing Metrics
 - Removed Herobrine