import numpy as np
import pandas as pd
from model import KnowledgeModel
from belief import Belief
import matplotlib.pyplot as plt
import os


class Simulator():
    """A simulator to run multiple knowledge models."""
    def __init__(self, N, S, T, graph, nx_params, sharingMode, shareTimeLimit, delay, singleSource):
        self.N = N
        self.S = S
        self.T = T
        self.graph = graph
        self.nx_params = nx_params
        self.mode = sharingMode
        self.delay = delay
        self.singleSource = singleSource
        self.shareTimeLimit = shareTimeLimit

    def runModel(self, network):
        """Run network model for T timesteps and return logs."""

        # create model
        model = KnowledgeModel(network=network,
                               sharingMode=self.mode,
                               sharetime=self.shareTimeLimit,
                               delay=self.delay,
                               singleSource=self.singleSource)

        # run model
        for t in range(self.T - 1):
            model.step()

        return model.logs()

    def runSimulation(self):
        """Run model over S simulations and return aggregate logs."""

        num_fake_per_agent = np.empty(shape=(self.S))
        neutral_per_timestep = np.empty(shape=(self.S, self.T))
        fake_per_timestep = np.empty(shape=(self.S, self.T))
        retracted_per_timestep = np.empty(shape=(self.S, self.T))

        for s in range(self.S):
            # run model
            network = self.graph(**self.nx_params)  # generate network from graph and params
            logs = self.runModel(network=network)
            df_belief = pd.DataFrame.from_dict(logs[0])

            # eval output
            num_fake_per_agent[s] = np.mean(np.sum(df_belief.values == Belief.Fake, axis=0))
            fake_per_timestep[s, :] = np.mean(df_belief.values == Belief.Fake, axis=1)
            retracted_per_timestep[s, :] = np.mean(df_belief.values == Belief.Retracted, axis=1)
            neutral_per_timestep[s, :] = np.mean(df_belief.values == Belief.Neutral, axis=1)

        # aggregate beliefs over time
        avg_fake_per_agent = np.mean(num_fake_per_agent)
        frac_neutral_per_timestep = np.mean(neutral_per_timestep, axis=0)
        frac_neutral_per_timestep_sd = np.std(neutral_per_timestep, axis=0)
        frac_fake_per_timestep = np.mean(fake_per_timestep, axis=0)
        frac_fake_per_timestep_sd = np.std(fake_per_timestep, axis=0)
        frac_retracted_per_timestep = np.mean(retracted_per_timestep, axis=0)
        frac_retracted_per_timestep_sd = np.std(retracted_per_timestep, axis=0)

        # aggregate final belief distributions
        neutral_dist = neutral_per_timestep[:, self.T-1]
        fake_dist = fake_per_timestep[:, self.T-1]
        retracted_dist = retracted_per_timestep[:, self.T-1]

        # bundle aggregated output
        frac_belief_mean = (frac_neutral_per_timestep, frac_fake_per_timestep, frac_retracted_per_timestep)
        frac_belief_sd = (frac_neutral_per_timestep_sd, frac_fake_per_timestep_sd, frac_retracted_per_timestep_sd)
        belief_dist = (neutral_dist, fake_dist, retracted_dist)

        return avg_fake_per_agent, frac_belief_mean, frac_belief_sd, belief_dist

    def visFinalBeliefDistributions(self, belief_dist, experiment, subexperiment, network_name, nx_params, save):
        """Plot belief distributions on final time step over all simulations."""

        neutral_dist, fake_dist, retracted_dist = belief_dist
        ranges = np.linspace(start=0, stop=1, num=100)

        plt.subplot(3, 1, 1)
        plt.hist(neutral_dist, bins=ranges)
        plt.ylim(ymin=0, ymax=self.S)
        plt.ylabel("Neutral")

        plt.subplot(3, 1, 2)
        plt.hist(fake_dist, bins=ranges)
        plt.ylim(ymin=0, ymax=self.S)
        plt.ylabel("Fake")

        plt.subplot(3, 1, 3)
        plt.hist(retracted_dist, bins=ranges)
        plt.ylim(ymin=0, ymax=self.S)
        plt.xlabel("Fraction of population holding belief at time T")
        plt.ylabel("Retracted")

        if save:  # write plot to output directory
            directory = "./output/" + experiment + "/" + subexperiment + "/hist/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            network_name = network_name + '_' + '_'.join(['{}={}'.format(k, v) for k, v in nx_params.items()])
            plt.savefig(directory + "/N{N}-T{T}-S{S}-{shr}-{dly}-{name}-hist.png".format(
                N=self.N, T=self.T, S=self.S, shr=self.shareTimeLimit, dly=self.delay, name=network_name), bbox_inches="tight")

        plt.show()

    def visBeliefsOverTime(self, data, experiment, subexperiment, network_name, nx_params, save, plot_sd=False):
        """Plot data and output to file."""

        # unpack aggregate data
        avg_num_fake, frac_belief_mean, frac_belief_sd = data
        neutral_mean, fake_mean, retracted_mean = frac_belief_mean
        neutral_sd, fake_sd, retracted_sd = frac_belief_sd

        alpha = 0.5
        plt.plot(range(self.T), fake_mean, label="False", color="tab:red", ls="-")
        plt.plot(range(self.T), neutral_mean, label="Neutral", color="tab:orange", ls="-")
        plt.plot(range(self.T), retracted_mean, label="Retracted", color="tab:green", ls="-")
        if plot_sd:
            plt.plot(range(self.T), fake_mean + fake_sd, color="tab:red", ls="--", alpha=alpha)
            plt.plot(range(self.T), fake_mean - fake_sd, color="tab:red", ls="--", alpha=alpha)
            plt.plot(range(self.T), neutral_mean + neutral_sd, color="tab:orange", ls="--", alpha=alpha)
            plt.plot(range(self.T), neutral_mean - neutral_sd, color="tab:orange", ls="--", alpha=alpha)
            plt.plot(range(self.T), retracted_mean + retracted_sd, color="tab:green", ls="--", alpha=alpha)
            plt.plot(range(self.T), retracted_mean - retracted_sd, color="tab:green", ls="--", alpha=alpha)
        plt.xlim(0, self.T)
        plt.ylim(0, 1.11)
        plt.xlabel("Time")
        plt.ylabel("Proportion of population holding belief")
        title_params = {"N":self.N, "T":self.T, "S":self.S, "avg":round(avg_num_fake, 1), "shr":self.shareTimeLimit, "dly":self.delay}
        plt.title("N = {N}, T = {T}, S = {S}, Num = {avg}, Share = {shr}, Delay = {dly}".format(**title_params))
        plt.legend(loc="lower center", ncol=3, fancybox=True, bbox_to_anchor=(0.5, 0.9))

        if save:  # write plot to output directory
            directory = "./output/" + experiment + "/" + subexperiment
            directory += "/sd" if plot_sd else ""  # create subfolder for sd plots
            if not os.path.exists(directory):
                os.makedirs(directory)
            network_name = network_name + '_' + '_'.join(['{}={}'.format(k, v) for k, v in nx_params.items()])
            plt.savefig(directory + "/N{N}-T{T}-S{S}-{shr}-{dly}-{name}-{avg}{sd}.png".format(
                N=self.N, T=self.T, S=self.S, shr=self.shareTimeLimit, dly=self.delay, name=network_name,
                avg=round(avg_num_fake, 1), sd=("-sd" if plot_sd else "")), bbox_inches="tight")

        plt.show()
