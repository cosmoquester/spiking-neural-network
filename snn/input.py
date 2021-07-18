import numpy as np


class RandomStimulator:
    def __init__(self, shape=()):
        self.shape = shape

    def __call__(self):
        return np.random.randint(0, 2, self.shape).tolist()
