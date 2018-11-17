from mesa import Model
from mesa.time import RandomActivation
from agent import PopAgent, Belief
from logger import Logger


class KnowledgeModel(Model):
    """A model with some number of agents."""

    def __init__(self, network):
        self.G = network
        self.num_agents = self.G.number_of_nodes()
        self.schedule = RandomActivation(self)
        self.logger = Logger(self)

        # Create agents
        for i in range(self.num_agents):
            neighbors = list(self.G.neighbors(i))
            a = PopAgent(unique_id=i, model=self, neighbors=neighbors)

            if i == 0:
                # Give Agent 0 false information
                a.belief = Belief.Fake
            if i == 1:
                # Give Agent 1 true information
                a.belief = Belief.Retracted
            self.schedule.add(a)
            self.logger.add(a)

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        self.logger.log()

