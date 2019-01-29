from model import KnowledgeModel
from belief import Belief
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Hyperparameters:
N = 1000     # Number of agents in the model
T = 10000    # Number of time steps per simulation
S = 100      # Number of simulations to run
sharetime = np.infty  # Time an agent will share newly attained beliefs; set np.infty for unlimited
delay = 0  # Time delay before retracted belief is added to model; set 0 for immediate addition
network = nx.complete_graph(N)   # Agent network
network.name = "complete"        # Network type used for output
experiment = "Delayed Retraction Model"  # Name of model (sets output folder and agent sharing constraints)
plot_sd = True  # show standard deviation on output plot
save = False  # write results to output folder

def runModel(name, network, T, delay):
    # create model
    model = KnowledgeModel(network=network, name=name, sharetime=sharetime, delay=delay)
    for t in range(T-1):
        model.step()
    return model.logs()

def runSimulation(S, T, network, experiment, delay):
    num_fake_per_agent = np.empty(shape=(S))
    fake_per_timestep = np.empty(shape=(S, T))
    retracted_per_timestep = np.empty(shape=(S, T))
    neutral_per_timestep = np.empty(shape=(S, T))

    for s in range(S):
        # run model
        logs = runModel(network=network, name=experiment, T=T, delay=delay)
        df_belief = pd.DataFrame.from_dict(logs[0])

        # eval output
        num_fake_per_agent[s] = np.mean(np.sum(df_belief.values == Belief.Fake, axis=0))
        fake_per_timestep[s,:] = np.mean(df_belief.values == Belief.Fake, axis=1)
        retracted_per_timestep[s,:] = np.mean(df_belief.values == Belief.Retracted, axis=1)
        neutral_per_timestep[s,:] = np.mean(df_belief.values == Belief.Neutral, axis=1)

    avg_num_fake_per_agent = np.mean(num_fake_per_agent)
    frac_fake_per_timestep = np.mean(fake_per_timestep, axis=0)
    frac_fake_per_timestep_sd = np.std(fake_per_timestep, axis=0)
    frac_retracted_per_timestep = np.mean(retracted_per_timestep, axis=0)
    frac_retracted_per_timestep_sd = np.std(retracted_per_timestep, axis=0)
    frac_neutral_per_timestep = np.mean(neutral_per_timestep, axis=0)
    frac_neutral_per_timestep_sd = np.std(neutral_per_timestep, axis=0)

    frac_belief_mean = (frac_neutral_per_timestep, frac_fake_per_timestep, frac_retracted_per_timestep)
    frac_belief_sd = (frac_neutral_per_timestep_sd, frac_fake_per_timestep_sd, frac_retracted_per_timestep_sd)

    return avg_num_fake_per_agent, frac_belief_mean, frac_belief_sd


avg_num_fake, frac_belief_mean, frac_belief_sd = runSimulation(S, T, network, experiment, delay)
neutral_mean, fake_mean, retracted_mean = frac_belief_mean
neutral_sd, fake_sd, retracted_sd = frac_belief_sd
print("Average number of time steps holding false belief:", avg_num_fake)


alpha = 0.5
plt.plot(range(T), fake_mean, label="False", color="tab:red", ls="-")
plt.plot(range(T), neutral_mean, label="Neutral", color="tab:orange", ls="-")
plt.plot(range(T), retracted_mean, label="Retracted", color="tab:green", ls="-")
if plot_sd:
    plt.plot(range(T), fake_mean+fake_sd, color="tab:red", ls="--", alpha=alpha)
    plt.plot(range(T), fake_mean-fake_sd, color="tab:red", ls="--", alpha=alpha)
    plt.plot(range(T), neutral_mean+neutral_sd, color="tab:orange", ls="--", alpha=alpha)
    plt.plot(range(T), neutral_mean-neutral_sd, color="tab:orange", ls="--", alpha=alpha)
    plt.plot(range(T), retracted_mean+retracted_sd, color="tab:green", ls="--", alpha=alpha)
    plt.plot(range(T), retracted_mean-retracted_sd, color="tab:green", ls="--", alpha=alpha)
plt.xlim(0,T)
plt.ylim(0,1.11)
plt.xlabel("Time")
plt.ylabel("Proportion of population holding belief")
plt.title("N = {N}, T = {T}, S = {S}, Num = {avg}, Share = {shr}, Delay = {dly}".format(N=N, T=T, S=S, avg=round(avg_num_fake, 1),
                                                                                        shr=sharetime, dly = delay))
plt.legend(loc="lower center", ncol=3, fancybox=True, bbox_to_anchor=(0.5, 0.9))
if save: # write plot to output directory
    directory = "./output/" + experiment + "/" + str(N)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "/N{N}-T{T}-S{S}-{shr}-{dly}-{name}-{avg}{sd}.png".format(
        N=N, T=T, S=S, shr=sharetime, dly=delay, name=network.name, avg=round(avg_num_fake,1), sd=("-sd" if plot_sd else "")), bbox_inches="tight")
plt.show()


# beliefs, interactions, pairs = runModel(network, T, debug=True)
# df_belief = pd.DataFrame.from_dict(beliefs)

# print(beliefs)       # id: belief.value
# print(interactions)  # id: [(interlocutor0_id, interlocutor0_belief), (interlocutor1_id, interlocutor1_belief), ...]
# print(pairs)         # [{agentA1, agentB1}, {agentA2, agentB2}, ..., {agentAT, agentBT}]
# print(df_belief)     # columns = ID, rows = belief at time step
