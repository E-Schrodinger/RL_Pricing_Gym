import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class Pricing_Metric:
    def __init__(self, **kwargs):
        self.iterations = kwargs.get('iterations', 100)

    def average_price(self, game, Agent1, Agent2):
        single_iter_average1 = 0
        single_iter_average2 = 0
        for _ in range(int(self.iterations)):
            Agent1.reset(game)
            Agent2.reset(game)
            game, s, all_visited_states, all_actions = game.simulate_game(Agent1, Agent2, game)
            single_iter_average1 += np.sum([game.init_actions()[a[0]] for a in all_actions[-int(game.tstable):]])/int(game.tstable)
            single_iter_average2 += np.sum([game.init_actions()[a[1]] for a in all_actions[-int(game.tstable):]])/int(game.tstable)
        
        print(f"Average price set by Agent 1 over {self.iterations} iterations = {single_iter_average1/self.iterations}")
        print(f"Average price set by Agent 2 over {self.iterations} iterations = {single_iter_average2/self.iterations}")

        return single_iter_average1/self.iterations, single_iter_average2/self.iterations
    
    def create_heatmap(self, state_counts_agent1, state_counts_agent2):
    # Create a 2D array for the heatmap
        heatmap_data = np.zeros((len(state_counts_agent2), len(state_counts_agent1)))
        
        # Normalize the data
        total_count = sum(state_counts_agent1) + sum(state_counts_agent2)
        for i in range(len(state_counts_agent2)):
            for j in range(len(state_counts_agent1)):
                heatmap_data[i, j] = (state_counts_agent1[j] + state_counts_agent2[i]) / total_count

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap_data, cmap='YlOrRd')

        # Set title and labels
        ax.set_title('State Distribution Heatmap')
        ax.set_xlabel('Agent 1 States')
        ax.set_ylabel('Agent 2 States')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Normalized State Count', rotation=-90, va="bottom")

        # Add text annotations
        for i in range(len(state_counts_agent2)):
            for j in range(len(state_counts_agent1)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                            ha="center", va="center", color="black")

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
    
    def state_heatmap(self, game, Agent1, Agent2):
        sdim, _ = game.init_state()
        state_counts_agent1 = [0] * sdim[0]
        state_counts_agent2 = [0] * sdim[0]

        for _ in range(int(self.iterations)):
            Agent1.reset(game)
            Agent2.reset(game)
            game, s, all_visited_states, all_actions = game.simulate_game(Agent1, Agent2, game)
            agent1_counted = [state[0] for state in all_visited_states[-int(game.tstable):]]
            agent2_counted = [state[1] for state in all_visited_states[-int(game.tstable):]]

            # Count occurrences for each agent
            count1 = Counter(agent1_counted)
            count2 = Counter(agent2_counted)

            # Update state_counts while maintaining the original length
            for i in range(sdim[0]):
                state_counts_agent1[i] += count1.get(i, 0)
                state_counts_agent2[i] += count2.get(i, 0)

        print(state_counts_agent1)
        print(state_counts_agent2)

        # print(agent1_counted)
        # print(agent2_counted)

        self.create_heatmap(state_counts_agent1, state_counts_agent2)

        return state_counts_agent1, state_counts_agent2




