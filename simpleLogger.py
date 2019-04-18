class SimpleLogger:
    """A simple logger tracking agents' belief and interaction history."""

    def __init__(self, model):
        self.model = model
        self.belief_history = dict()  # unique_id: belief.value
        self.interaction_history = dict()  # unique_id: [(interlocutor_id, interlocutor.belief)]
        self.pair_history = []   # [{agentA, agentB}, {agentA, agentB}]

    def add(self, agent):
        """Add agent to logger"""
        self.belief_history[agent.unique_id] = [agent.belief.value]
        self.interaction_history[agent.unique_id] = []

    def logs(self):
        """Get agents' belief, interaction and pair history, respectively."""
        return self.belief_history, self.interaction_history, self.pair_history

    def log(self, agentA, agentB):
        """Log agents to belief and interaction history"""
        agents = self.model.schedule.agents

        # log belief history
        for agent in agents:
            self.belief_history[agent.unique_id].append(agent.belief.value)

        # log interaction history
        entryA = (agentA.unique_id, agentA.belief)
        entryB = (agentB.unique_id, agentB.belief)
        self.interaction_history[agentA.unique_id].append(entryB)
        self.interaction_history[agentB.unique_id].append(entryA)

        # log pair history
        pair = set((agentA.unique_id, agentB.unique_id))
        self.pair_history.append(pair)
