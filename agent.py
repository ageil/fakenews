from mesa import Agent
from belief import Belief
import random


class PopAgent(Agent):
    """An Agent with some initial knowledge."""

    def __init__(self, unique_id, model, neighbors):
        super().__init__(unique_id, model)

        # Default params
        self.belief = Belief.Neutral
        self.neighbors = neighbors  # list of agent's neighbours

    def interact(self):
        """Interact with neighbor and update agents' beliefs"""
        # Pick a neighbouring agent to interact with
        other_id = random.choice(self.neighbors)
        other = self.model.schedule.agents[other_id]

        # Ensure self and other are different
        while self.unique_id == other.unique_id:
            other = random.choice(self.neighbors)
        assert self.unique_id != other.unique_id, "Agents are not distinct!"

        self.model.logger.loginteraction(self, other)
        self.update(other)

    def update(self, other):
        """Update agents' beliefs"""
        # Convert self to false belief
        if self.belief == Belief.Neutral and other.belief == Belief.Fake:
            self.belief = Belief.Fake

        # Convert self to retracted belief
        if self.belief == Belief.Fake and other.belief == Belief.Retracted:
            self.belief = Belief.Retracted

        # Convert other to false belief
        if self.belief == Belief.Fake and other.belief == Belief.Neutral:
            other.belief = Belief.Fake

        # Convert other to retracted belief
        if self.belief == Belief.Retracted and other.belief == Belief.Fake:
            other.belief = Belief.Retracted

    def step(self):
        """Execute step"""
        self.interact()
