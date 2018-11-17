from mesa import Agent
from belief import Belief
import random


class PopAgent(Agent):
    """An Agent with some initial knowledge."""

    def __init__(self, unique_id, model, neighbors):
        super().__init__(unique_id, model)

        # Set every agent to no information at the outset
        self.belief = Belief.Neutral

        # Store list of agents neighbours
        self.neighbors = neighbors

    def update(self, other):
        """Update agent's own beliefs"""
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
        # During each step, we pick a neighbouring agent to interact with self
        other_id = random.choice(self.neighbors)
        other = self.model.schedule.agents[other_id]

        # Ensure self and other are different
        while self.unique_id == other.unique_id:
            other = random.choice(self.neighbors)

        # Sanity check, to ensure the agents are unique by this point
        assert self.unique_id != other.unique_id, "Something is awry!"

        # print([a.unique_id for a in agents])
        print("Interacting agents:", self.unique_id, other.unique_id)

        self.update(other)
