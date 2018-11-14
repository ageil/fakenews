from mesa import Model
from mesa.time import RandomActivation
from agent import PopAgent, Belief


class KnowledgeModel(Model):
    """A model with some number of agents."""

    def __init__(self, N):
        self.num_agents = N
        self.schedule = RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = PopAgent(i, self)
            if i == 0:
                # Give Agent 0 false information
                a.belief = Belief.Fake
            if i == 1:
                # Give Agent 1 true information
                a.belief = Belief.Retracted
            self.schedule.add(a)

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()