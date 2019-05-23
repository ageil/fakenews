from mesa import Model
from agent import PopAgent, Belief
from simpleScheduler import SimpleActivation
import random


class KnowledgeModel(Model):
    """A model with some number of agents."""

    def __init__(self, network, sharingMode, sharetime, delay, singleSource=False, samePartition=True):
        self.mode = sharingMode
        self.G = network
        self.num_agents = self.G.number_of_nodes()
        self.schedule = SimpleActivation(self)
        self.delay = delay
        self.singleSource = singleSource
        self.samePartition = samePartition

        # Create agents
        for i in range(self.num_agents):
            neighbors = list(self.G.neighbors(i))
            a = PopAgent(unique_id=i, model=self, neighbors=neighbors, sharetime=sharetime)

            if i == 0:
                # Give Agent 0 false information
                a.belief = Belief.Fake
                self.agentZero = a
            if (i == 1) and (self.delay == 0):
                # Give Agent 1 true information
                a.belief = Belief.Retracted
            self.schedule.add(a)

    def addRetracted(self):
        """Add retracted belief to random agent."""
        if self.singleSource:
            a = self.agentZero
        elif 'partition' in self.G.graph.keys() and self.samePartition:
            num = random.choice(list(self.G.graph['partition'][0]))
            a = self.schedule.agents[num]
        elif 'partition' in self.G.graph.keys() and not self.samePartition:
            num = random.choice(list(self.G.graph['partition'][1]))
            a = self.schedule.agents[num]
        else:
            a = random.choice(self.schedule.agents)
        a.setBelief(Belief.Retracted)

    def step(self):
        """Advance the model by one step."""
        if (self.delay > 0) and (self.schedule.time == self.delay):
            self.addRetracted()

        self.schedule.step()

    def logs(self):
        """Get agents' belief, interaction and pair history, respectively."""
        return self.schedule.logs()