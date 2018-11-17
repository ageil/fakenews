from model import KnowledgeModel
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


# Number of agents in the model
N = 3

# Number of iterations
timesteps = 4

# create network
network = nx.complete_graph(N)

# create model
model = KnowledgeModel(network)

for timestep in range(timesteps):
    print()
    print("~~~~~ ROUND " + str(timestep) + " ~~~~~")
    print()

    for agent in model.schedule.agents:
        print(agent.unique_id, agent.belief)

    model.step()

# belief output
hist_belief = model.logger.belief_history
df_belief = pd.DataFrame.from_dict(hist_belief)

# interaction output
hist_action = model.logger.interaction_history
df_action = pd.DataFrame.from_dict(hist_action)

print(df_belief)
print(df_action)

# agent_knowledge = [a.belief for a in model.schedule.agents]
# plt.hist(agent_knowledge)
# plt.show()
