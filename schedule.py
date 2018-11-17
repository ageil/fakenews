import random

class BaseScheduler:
    """ Simplest scheduler; activates agents one at a time, in the order
    they were added.

    Assumes that each agent added has a *step* method which takes no arguments.

    (This is explicitly meant to replicate the scheduler in MASON).

    """
    def __init__(self, model):
        """ Create a new, empty BaseScheduler. """
        self.model = model
        self.steps = 0
        self.time = 0
        self.agents = []

    def add(self, agent):
        """ Add an Agent object to the schedule.

        Args:
            agent: An Agent to be added to the schedule. NOTE: The agent must
            have a step() method.

        """
        self.agents.append(agent)


    def remove(self, agent):
        """ Remove all instances of a given agent from the schedule.

        Args:
            agent: An agent object.

        """
        while agent in self.agents:
            self.agents.remove(agent)


    def step(self):
        """ Execute the step of all the agents, one at a time. """
        for agent in self.agents[:]:
            agent.step()
        self.steps += 1
        self.time += 1


    def get_agent_count(self):
        """ Returns the current number of agents in the queue. """
        return len(self.agents)

class SingleRandomActivation(BaseScheduler):

    def step(self):
        """ Executes the step of all agents, one at a time, in
        random order.

        """
        random.shuffle(self.agents)
        for agent in self.agents[:1]:
            agent.step()
        self.steps += 1
        self.time += 1
