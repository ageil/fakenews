from mesa import Model
from agent import PopAgent, Belief
from simpleScheduler import SimpleActivation
import random


class KnowledgeModel(Model):
    """A model with some number of agents."""

    def __init__(self, network, name, sharetime, delay):
        self.name = name
        self.G = network
        self.num_agents = self.G.number_of_nodes()
        self.schedule = SimpleActivation(self)
        self.delay = delay

        # Create agents
        for i in range(self.num_agents):
            neighbors = list(self.G.neighbors(i))
            a = PopAgent(unique_id=i, model=self, neighbors=neighbors, sharetime=sharetime)

            if i == 0:
                # Give Agent 0 false information
                a.belief = Belief.Fake
            if (i == 1) and (self.delay == 0):
                # Give Agent 1 true information
                a.belief = Belief.Retracted
            self.schedule.add(a)

    def addRetracted(self):
        """Add retracted belief to random agent."""
        a = random.choice(self.schedule.agents)
        a.belief = Belief.Retracted

    def step(self):
        """Advance the model by one step."""
        if (self.delay > 0) and (self.schedule.time == self.delay):
            self.addRetracted()

        self.schedule.step()

    def logs(self):
        """Get agents' belief, interaction and pair history, respectively."""
        return self.schedule.logs()