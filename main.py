from model import KnowledgeModel
from belief import Belief
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters:
S = 1000  # Number of simulations to run
T = 100  # Number of time steps per simulation
N = 10   # Number of agents in the model
network = nx.complete_graph(N)   # Agent network
network.name = "complete"        # Network type used for output


def runModel(network, T, debug=False):
    # create model
    model = KnowledgeModel(network)

    for t in range(T-1):
        if debug:
            print("\n~~~~~ ROUND " + str(t) + " ~~~~~\n")
            for agent in model.schedule.agents:
                print(agent.unique_id, agent.belief)
        model.step()

    return model.logs()


def runSimulation(S, T, network):
    N = network.number_of_nodes()
    num_fake_per_agent = np.empty(shape=(S))
    fake_per_timestep = np.empty(shape=(S, T))
    retracted_per_timestep = np.empty(shape=(S, T))
    neutral_per_timestep = np.empty(shape=(S, T))

    for s in range(S):
        # run model
        logs = runModel(network, T)
        df_belief = pd.DataFrame.from_dict(logs[0])

        # eval output
        num_fake_per_agent[s] = np.mean(np.sum(df_belief.values == Belief.Fake, axis=0))
        fake_per_timestep[s,:] = np.mean(df_belief.values == Belief.Fake, axis=1)
        retracted_per_timestep[s,:] = np.mean(df_belief.values == Belief.Retracted, axis=1)
        neutral_per_timestep[s,:] = np.mean(df_belief.values == Belief.Neutral, axis=1)

    avg_num_fake_per_agent = np.mean(num_fake_per_agent)
    frac_fake_per_timestep = np.mean(fake_per_timestep, axis=0)
    frac_retracted_per_timestep = np.mean(retracted_per_timestep, axis=0)
    frac_neutral_per_timestep = np.mean(neutral_per_timestep, axis=0)

    return avg_num_fake_per_agent, frac_fake_per_timestep, frac_retracted_per_timestep, frac_neutral_per_timestep


avg_num_fake, frac_fake, frac_rtrt, frac_ntrl = runSimulation(S, T, network)
print("Average number of time steps holding false belief:", avg_num_fake)

plt.plot(range(T), frac_fake, label="False", color="tab:red")
plt.plot(range(T), frac_ntrl, label="Neutral", color="tab:orange")
plt.plot(range(T), frac_rtrt, label="Retracted", color="tab:green")
plt.xlim(0,T)
plt.ylim(0,1.11)
plt.xlabel("Time")
plt.ylabel("Proportion of population holding belief")
plt.title("N = {N}, T = {T}, S = {S}, Num = {avg}".format(N=N, T=T, S=S, avg=round(avg_num_fake, 1)))
plt.legend(loc="lower center", ncol=3, fancybox=True, bbox_to_anchor=(0.5, 0.9))
plt.savefig("./output/S{S}-T{T}-N{N}-{avg}-{name}.png".format(S=S, T=T, N=N, avg=round(avg_num_fake,1), name=network.name), bbox_inches="tight")
plt.show()


# beliefs, interactions, pairs = runModel(network, T, debug=True)
# df_belief = pd.DataFrame.from_dict(beliefs)

# print(beliefs)       # id: belief.value
# print(interactions)  # id: [(interlocutor0_id, interlocutor0_belief), (interlocutor1_id, interlocutor1_belief), ...]
# print(pairs)         # [{agentA1, agentB1}, {agentA2, agentB2}, ..., {agentAT, agentBT}]
# print(df_belief)     # columns = ID, rows = belief at time step
