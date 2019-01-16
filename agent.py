from mesa import Agent
from belief import Belief


class PopAgent(Agent):
    """An Agent with some initial knowledge."""

    def __init__(self, unique_id, model, neighbors, num_shares):
        super().__init__(unique_id, model)

        # Default params
        self.belief = Belief.Neutral
        self.neighbors = neighbors  # list of agent's neighbours
        self.num_shares = num_shares

    def update(self, other):
        """Update agent's own beliefs"""
        isSharing = other.num_shares > 0

        # Convert self to false belief
        if self.belief == Belief.Neutral and other.belief == Belief.Fake and isSharing:
            self.belief = Belief.Fake
            other.num_shares -= 1

        # Convert self to retracted belief
        if self.belief == Belief.Fake and other.belief == Belief.Retracted and isSharing:
            self.belief = Belief.Retracted
            other.num_shares -= 1

    def step(self, interlocutor):
        """Interact with interlocutor, updating own beliefs."""
        self.update(interlocutor)
