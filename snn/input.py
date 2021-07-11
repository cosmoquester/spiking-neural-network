import random


class RandomStimulator:
    def __call__(self):
        return random.randrange(0, 2)
