from typing import List, Tuple

from .configs import NeuronConfig


class Neuron:
    def __init__(self, cfg: NeuronConfig):
        """
        Initialize neuron

        :param cfg: neuron config instance
        """
        self.rest_potential = cfg.rest_potential
        self.min_potential = cfg.min_potential
        self.threshold_potential = cfg.threshold_potential
        self.excitatory_potential = cfg.excitatory_potential
        self.leakage_factor = cfg.leakage_factor
        self.rest_period = cfg.rest_period

        self.reset()

    def __call__(self, input_potential: float) -> Tuple[bool, float]:
        """
        Input potentials to this neuron

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

    def reset(self):
        """Reset neuron's status"""
        self.rest = 0
        self.potential = self.rest_potential


class Neurons:
    """A bunch of Neurons of same parameters"""

    def __init__(self, num_neurons: int, cfg: NeuronConfig):
        self.neurons = [Neuron(cfg) for _ in range(num_neurons)]

    def __call__(self, input_potentials: List[float]) -> Tuple[List[bool], List[float]]:
        """
        Input potentials to each neuron and return spikes and potentials

        :param input_potentials: input potentials for each neuron
        :returns: tuple of `is firing` list and `potential` list of each neuron
        """
        assert len(input_potentials) == len(
            self.neurons
        ), "The number of potential is different from the number of neurons!"

        fire_spikes = []
        output_potentials = []
        for input_potential, neuron in zip(input_potentials, self.neurons):
            fire_spike, output_potential = neuron(input_potential)
            fire_spikes.append(fire_spike)
            output_potentials.append(output_potential)

        return fire_spikes, output_potentials

    def reset(self):
        """Reset neuron's status"""
        for neuron in self.neurons:
            neuron.reset()
