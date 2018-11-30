from model import KnowledgeModel
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


# Number of agents in the model
N = 3

# Number of time steps to simulate
T = 4

# create network
network = nx.complete_graph(N)

# create model
model = KnowledgeModel(network)

for t in range(T):
    print("\n~~~~~ ROUND " + str(t) + " ~~~~~\n")
    for agent in model.schedule.agents:
        print(agent.unique_id, agent.belief)
    model.step()

# belief output
beliefs, interactions, pairs = model.logs()
df_belief = pd.DataFrame.from_dict(beliefs)

print(df_belief)     # columns = ID, rows = belief at timestep
print(interactions)  # id: [(interlocutor0_id, interlocutor0_belief), (interlocutor1_id, interlocutor1_belief), ...]
print(pairs)         # [{agentA1, agentB1}, {agentA2, agentB2}, ..., {agentAT, agentBT}]