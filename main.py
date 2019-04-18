import numpy as np
import networkx as nx
from belief import Mode
from simulator import Simulator

# Hyperparameters
N = 100     # Number of agents in the network
T = 1000    # Number of time steps per simulation
S = 1000      # Number of simulations to run

# Agent belief sharing constraints
mode = Mode.CorrectionFatigue   # Set agent sharing mode
shareTimeLimit = np.infty       # Time an agent will share their newly attained beliefs; set np.infty for unlimited

# Delayed introduction of retracted belief
delay = 0             # Time delay before retracted belief is added to model; set 0 for immediate addition
singleSource = False  # Retracted source same as false belief source (False only applied if delay > 0)

# Graph & network structure
graph = nx.complete_graph
nx_params = {"n": N}
# graph = nx.watts_strogatz_graph
# nx_params = {"n": N, "k": 32, "p": 0.16}
# graph = nx.random_partition_graph                          # Agent graph function
# nx_params = {"sizes": [50,50], "p_in": 0.4, "p_out": 0.2}  # Graph parameters

# Output & naming
experiment = "Correction Fatigue Model"      # Set output folder name
subexperiment = "100"             # Set output subfolder name
network_name = "complete"    # Set network name for output file
save = True                         # Write plot to file

# Run simulator
sim = Simulator(N=N, S=S, T=T,
                graph=graph,
                nx_params=nx_params,
                sharingMode=mode,
                shareTimeLimit=shareTimeLimit,
                delay=delay,
                singleSource=singleSource)
data = sim.runSimulation(save=save,
                         experiment=experiment,
                         subexperiment=subexperiment,
                         network_name=network_name,
                         nx_params=nx_params)
avg, frac_mean, frac_sd, belief_dist = data
temporal_data = avg, frac_mean, frac_sd

# Visualize output
sim.visBeliefsOverTime(data=temporal_data,
                       experiment=experiment,
                       subexperiment=subexperiment,
                       network_name=network_name,
                       nx_params=nx_params,
                       save=save,
                       plot_sd=False)
sim.visBeliefsOverTime(data=temporal_data,
                       experiment=experiment,
                       subexperiment=subexperiment,
                       network_name=network_name,
                       nx_params=nx_params,
                       save=save,
                       plot_sd=True)
sim.visFinalBeliefDistributions(belief_dist=belief_dist,
                                experiment=experiment,
                                subexperiment=subexperiment,
                                network_name=network_name,
                                nx_params=nx_params,
                                save=save)


# beliefs, interactions, pairs = runModel(network, T, debug=True)
# df_belief = pd.DataFrame.from_dict(beliefs)

# print(beliefs)       # id: belief.value
# print(interactions)  # id: [(interlocutor0_id, interlocutor0_belief), (interlocutor1_id, interlocutor1_belief), ...]
# print(pairs)         # [{agentA1, agentB1}, {agentA2, agentB2}, ..., {agentAT, agentBT}]
# print(df_belief)     # columns = ID, rows = belief at time step
