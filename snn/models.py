from typing import Tuple


class Neuron:
    def __init__(
        self,
        rest_potential: float,
        min_potential: float,
        threshold_potential: float,
        excitatory_potential: float,
        leakage_factor: float,
        rest_period: int,
    ):
        """
        Initialize neuron

        :param rest_potential: potential when resting
        :param min_potential: minimum potential
        :param threshold_potential: threshold that fire a spike when potential is over than
        :param excitatory_potential: added potential when excited
        :param leakage_factor: decreasing potential by leakage_factor
        :param rest_period: the number of time step in which neuron can't be receive input potential
        """
        self.rest_potential = rest_potential
        self.min_potential = min_potential
        self.threshold_potential = threshold_potential
        self.excitatory_potential = excitatory_potential
        self.leakage_factor = leakage_factor
        self.rest_period = rest_period

        self.potential = rest_potential
        self.rest = 0

    def __call__(self, input_potential: float) -> Tuple[bool, float]:
        """
        Input spike to this neuron or not

        :param input_potential: input potential
        :returns: tuple of `is firing` and `potential of neuron`
        """
        if self.rest:
            self.rest -= 1
            self.potential = self.rest_potential
            return False, self.potential

        if self.potential > self.min_potential:
            self.potential += input_potential - self.leakage_factor
        else:
            self.potential = 0.0

        fire_spike = self.potential > self.threshold_potential
        if fire_spike:
            self.potential += self.excitatory_potential
            self.rest += self.rest_period
        return fire_spike, self.potential
