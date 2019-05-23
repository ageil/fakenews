from mesa import Agent
from belief import Belief, Mode


class PopAgent(Agent):
    """An Agent with some initial knowledge."""

    def __init__(self, unique_id, model, neighbors, sharetime):
        super().__init__(unique_id, model)

        # Default params
        self.belief = Belief.Neutral
        self.neighbors = neighbors  # list of agent's neighbours
        self.clock = 0  # internal timer (absolute time)
        self.beliefTime = 0  # time current belief has been held
        self.shareTime = sharetime   # time limit within which new beliefs are shared

    def tick(self):
        """Increment clock by 1."""
        self.clock += 1
        self.beliefTime += 1

    def isSharing(self):
        """Check if agent is still sharing own belief."""
        return self.beliefTime <= self.shareTime

    def setBelief(self, belief):
        """Set agent's belief."""
        self.belief = belief
        self.beliefTime = 0

    def update(self, other):
        """Update agent's own beliefs"""

        # Check if other is sharing belief (model dependent)
        if self.model.mode == Mode.TimedNovelty:
            isSharingFake = other.isSharing()
            isSharingRetracted = other.isSharing()
        elif self.model.mode == Mode.CorrectionFatigue:
            isSharingFake = True
            isSharingRetracted = other.isSharing()
        else:
            isSharingFake = True
            isSharingRetracted = True

        # Convert self to false belief
        if self.belief == Belief.Neutral and other.belief == Belief.Fake and isSharingFake:
            self.setBelief(Belief.Fake)

        # Convert self to retracted belief
        if self.belief == Belief.Fake and other.belief == Belief.Retracted and isSharingRetracted:
            self.setBelief(Belief.Retracted)

    def step(self, interlocutor):
        """Interact with interlocutor, updating own beliefs."""
        self.update(interlocutor)
