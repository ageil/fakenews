from mesa import Model


class Logger(Model):
    """A model with some number of agents."""

    def __init__(self, model):
        self.model = model
        self.belief_history = dict()

    def add(self, agent):
        """Add agent to logger"""
        self.belief_history[agent.unique_id] = [agent.belief.value]

    def remove(self, agent):
        """Remove agent from logger"""
        self.belief_history.pop(agent.unique_id)

    def update(self):
        """Add agent beliefs to belief history"""
        agents = self.model.schedule.agents

        for agent in agents:
            self.belief_history[agent.unique_id].append(agent.belief.value)
