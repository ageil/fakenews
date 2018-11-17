from model import KnowledgeModel
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


# Number of agents in the model
N = 4

# Number of iterations
timesteps = 3

# type of network
network = nx.complete_graph(N)

# create model
model = KnowledgeModel(network)

for i in range(timesteps):
    print("~~~~~ ROUND " + str(i) + " ~~~~~")
    print("")
    for agent in model.schedule.agents:
        print(agent.unique_id, agent.belief)
    model.step()

# log output
hist = model.logger.belief_history
df = pd.DataFrame.from_dict(hist)

print(hist)
print(df)

#agent_knowledge = [a.belief for a in model.schedule.agents]
#plt.hist(agent_knowledge)
#plt.show()
