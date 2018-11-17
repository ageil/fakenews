from mesa import Model


class Logger(Model):
    """A network model with some number of agents"""

    def __init__(self, model):
        self.model = model
        self.belief_history = dict()  # unique_id: belief.value
        self.interaction_history = dict()  # timestep: [interactions]

    def add(self, agent):
        """Add agent to logger"""
        self.belief_history[agent.unique_id] = [agent.belief]
        self.interaction_history[agent.unique_id] = []

    def log(self):
        """Add agent beliefs to belief history"""
        agents = self.model.schedule.agents

        for agent in agents:
            ID = agent.unique_id
            self.belief_history[ID].append(agent.belief)

    def loginteraction(self, agent, other):
        self.interaction_history[agent.unique_id].append(other.unique_id)