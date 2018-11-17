from mesa.time import BaseScheduler
from simpleLogger import SimpleLogger
import random


class SimpleActivation(BaseScheduler):
    """ A simple scheduler which randomly picks only one pair of agents,
    activating their step function."""

    def __init__(self, model):
        super().__init__(model)

        # create logger
        self.logger = SimpleLogger(model)

    def add(self, agent):
        """Add an Agent object to the schedule and logger."""
        self._agents[agent.unique_id] = agent
        self.logger.add(agent)

    def logs(self):
        """Get agents' belief and interaction history, respectively."""
        return self.logger.logs()

    def choose(self):
        # pick agent A
        keys = list(self._agents.keys())
        keyA = random.choice(keys)
        agentA = self.model.schedule.agents[keyA]

        # pick pick agent B
        keyB = random.choice(agentA.neighbors)
        agentB = self.model.schedule.agents[keyB]

        return agentA, agentB

    def step(self):
        """Executes the step of one pair of agents."""
        # choose agent pair
        agentA, agentB = self.choose()

        # interact
        agentA.step(agentB)
        agentB.step(agentA)

        # log results
        self.logger.log(agentA, agentB)

        # increment counters
        self.steps += 1
        self.time += 1
