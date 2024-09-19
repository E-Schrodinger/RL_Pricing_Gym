import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

class Pricing_Metric:
    def __init__(self, game, Agent1, Agent2, **kwargs):
        self.iterations = kwargs.get('iterations', 100)
        self.simulation_results = None
        self.env = game
        self.Agent1 = Agent1
        self.Agent2 = Agent2

        self.Q_vals_1 = None
        self.Q_vals_2 = None


    def has_q_vals(self, agent):
        return (type(agent).__name__ in ['Q_Learning', 'Batch_SARSA'] or
                any(base.__name__ in ['Q_Learning', 'Batch_SARSA'] for base in type(agent).__bases__))

    def run_simulations(self):
        if self.simulation_results is None:
            self.simulation_results = []
            self.Q_vals_1 = []
            self.Q_vals_2 = []

            self.agent1_is_q = self.has_q_vals(self.Agent1)
            self.agent2_is_q = self.has_q_vals(self.Agent2)


            if (type(self.Agent1).__name__ in ['Q_Learning', 'Batch_SARSA'] or 
                any(base.__name__ in ['Q_Learning', 'Batch_SARSA'] for base in type(self.Agent1).__bases__)):

                for _ in range(int(self.iterations)):
                    self.Agent1.reset(self.env)
                    self.Agent2.reset(self.env)
                    self.env, s, all_visited_states, all_actions = self.env.simulate_game(self.Agent1, self.Agent2, self.env)
                    self.simulation_results.append((all_visited_states, all_actions))

                    if self.agent1_is_q:
                        self.Q_vals_1.append(self.Agent1.Q[0].copy())
                    else:
                        self.Q_vals_1.append(None)

                    if self.agent2_is_q:
                        self.Q_vals_2.append(self.Agent2.Q[0].copy())
                    else:
                        self.Q_vals_2.append(None)

    def average_price(self):
        self.run_simulations()

        single_iter_average1 = 0
        single_iter_average2 = 0
        for all_visited_states, all_actions in self.simulation_results:
            single_iter_average1 += np.sum([self.env.init_actions()[a[0]] for a in all_actions[-int(self.env.tstable):]])/int(self.env.tstable)
            single_iter_average2 += np.sum([self.env.init_actions()[a[1]] for a in all_actions[-int(self.env.tstable):]])/int(self.env.tstable)
        
        avg_price1 = single_iter_average1 / self.iterations
        avg_price2 = single_iter_average2 / self.iterations

        print(f"Average price set by Agent 1 over {self.iterations} iterations = {avg_price1}")
        print(f"Average price set by Agent 2 over {self.iterations} iterations = {avg_price2}")

        return avg_price1, avg_price2
    


    def create_heatmap(self, joint_state_keys, joint_state_counts):
        # Find the maximum state indices
        max_state1 = max(key[0] for key in joint_state_keys) + 1
        max_state2 = max(key[1] for key in joint_state_keys) + 1

        # Create a 2D array for the heatmap
        heatmap_data = np.zeros((max_state2, max_state1))

        # Fill the heatmap data
        for key, count in zip(joint_state_keys, joint_state_counts):
            heatmap_data[key[1], key[0]] = count

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap_data, cmap='YlOrRd')

        # Set title and labels
        ax.set_title('Joint State Distribution Heatmap')
        ax.set_xlabel('Agent 1 States')
        ax.set_ylabel('Agent 2 States')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('State Count', rotation=-90, va="bottom")

        # Add text annotations
        for i in range(max_state2):
            for j in range(max_state1):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.0f}',
                            ha="center", va="center", color="black")

        # Set ticks
        ax.set_xticks(np.arange(max_state1))
        ax.set_yticks(np.arange(max_state2))
        ax.set_xticklabels(np.arange(max_state1))
        ax.set_yticklabels(np.arange(max_state2))

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
    

    def state_heatmap(self):
        self.run_simulations()

        sdim, _ = self.env.init_state()

        joint_state_counts = [0] * (sdim[0] * sdim[0])
        joint_state_keys = [(i, j) for i in range(sdim[0]) for j in range(sdim[0])]

        for all_visited_states, _ in self.simulation_results:
            joint_counted = all_visited_states[-int(self.env.tstable):]
            joint_count = Counter(joint_counted)
            
            for i, key in enumerate(joint_state_keys):
                joint_state_counts[i] += joint_count.get(key, 0)

        self.create_heatmap(joint_state_keys, joint_state_counts)

        return joint_state_counts
    
    def Make_Q(self, index):
        if index == 0:
            Qvals = self.Q_vals_1
        elif index == 1:
            Qvals = self.Q_vals_2
        mean_Q = np.mean(Qvals, axis = 0)

        sdim, _ = self.env.init_state()

        column_names = []

        state_indices = []
        for i in range(sdim[0]):
                for j in range(sdim[0]):
                    state_indices.append((i,j))
        agent1_states = [state[0] for state in state_indices]
        agent2_states = [state[1] for state in state_indices]

        array_list = [agent1_states,agent2_states]

        for i in range(self.env.n):
            column_names.append(f"Agent {i+1} State")

        if index == 1:
            column_names.reverse()

        for k in range(mean_Q.shape[2]):
            column_names.append(f"{k}")

            action_list = []

            for i in range(sdim[0]):
                for j in range(sdim[0]):
                    action_list.append(mean_Q[(i,j)][k])
            
            array_list.append(action_list)
        
        for name in(array_list):
            print(len(name))
        data_dict = {name: arr for name, arr in zip(column_names, array_list)}
        df = pd.DataFrame(data_dict)


        return df

    def Print_Q(self, index = 0):
        if index == 0:
            Agent = self.Agent1
        elif index == 1:
            Agent = self.Agent2
        
        if self.has_q_vals(Agent):
            generated_Q_vals = self.Make_Q(index)
            print(generated_Q_vals)
            return generated_Q_vals
        else:
            print(f"Agent {index + 1} has no Q Values")
            return None
        
    def normalized_delta(self):
        p_competitive, p_monopoly = self.env.compute_p_competitive_monopoly()
        avg_price1, avg_price2 = self.average_price()
        pi_hat = (avg_price1+avg_price2)/self.env.n

        return (pi_hat - p_competitive)/(p_monopoly-p_competitive)





