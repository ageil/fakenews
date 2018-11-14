from mesa import Agent
from belief import Belief
import random


class PopAgent(Agent):
    """An Agent with some initial knowledge."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Set every agent to no information at the outset
        self.belief = Belief.Neutral

    def update(self, other):
        """Update agent's own beliefs"""
        # Convert to false belief
        if self.belief == Belief.Neutral and other.belief == Belief.Fake:
            self.belief = Belief.Fake

        # Retract false belief
        if self.belief == Belief.Fake and other.belief == Belief.Retracted:
            self.belief = Belief.Retracted

    def step(self):
        agents = self.model.schedule.agents

        # During each step, we pick an agent to interact with self.
        other = random.choice(agents)

        # Check to make sure self and other are unique
        #  If not, sample again (until unique agent found)
        while other.unique_id == self.unique_id:
            other = random.choice(agents)

        # Sanity check, to ensure the agents are unique by this point.
        if other.unique_id == self.unique_id:
            assert False, "Something is awry!"

        print("Interacting agents:", self.unique_id, other.unique_id)

        self.update(other)

# if self.belief + other.knowledge == 1:
# 	self.belief = 1
# 	other.knowledge = 1
# 	print("Fake news has spread")
# 	print("Agent-A is: " + str(self.unique_id))
# 	print("Agent-A knows: " + str(self.belief))
# 	print("Agent-B is: " + str(other.unique_id))
# 	print("Agent-B knows: " + str(other.knowledge))
# elif self.belief + other.knowledge == 3:
# 	self.belief = 2
# 	other.knowledge = 2
# 	print("agents have seen reason!")
# 	print("Agent-A is: " + str(self.unique_id))
# 	print("Agent-A knows: " + str(self.belief))
# 	print("Agent-B is: " + str(other.unique_id))
# 	print("Agent-B knows: " + str(other.knowledge))
# print("")
