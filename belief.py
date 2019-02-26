from enum import Enum


class Belief(Enum):
    Neutral = 0
    Fake = 1
    Retracted = 2

class Mode(Enum):
    Default = 0
    TimedNovelty = 1
    CorrectionFatigue = 2