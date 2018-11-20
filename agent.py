from mesa import Agent
from belief import Belief


class PopAgent(Agent):
    """An Agent with some initial knowledge."""

    def __init__(self, unique_id, model, neighbors):
        super().__init__(unique_id, model)

        # Default params
        self.belief = Belief.Neutral
        self.neighbors = neighbors  # list of agent's neighbours

    def update(self, other):
        """Update agent's own beliefs"""
        # Convert self to false belief
        if self.belief == Belief.Neutral and other.belief == Belief.Fake:
            self.belief = Belief.Fake

        # Convert self to retracted belief
        if self.belief == Belief.Fake and other.belief == Belief.Retracted:
            self.belief = Belief.Retracted

    def step(self, interlocutor):
        """Interact with interlocutor, updating own beliefs."""
        self.update(interlocutor)
