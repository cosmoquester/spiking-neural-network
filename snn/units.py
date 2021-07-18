from typing import Callable, List, Tuple

import numpy as np

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
        neuron_potentials = []
        for input_potential, neuron in zip(input_potentials, self.neurons):
            fire_spike, neuron_potential = neuron(input_potential)
            fire_spikes.append(fire_spike)
            neuron_potentials.append(neuron_potential)

        return fire_spikes, neuron_potentials

    def reset(self):
        """Reset neuron's status"""
        for neuron in self.neurons:
            neuron.reset()


class Synapse:
    """A bunch of synapse weights connecting multiple presynaptic neurons to postsynaptic neurons"""

    def __init__(self, num_presynaptic_neurons: int, num_postsynaptic_neurons: int, initializer: Callable = None):
        """
        Initialze synapse

        :param num_presynaptic_neurons: the number of presynaptic neurons
        :param num_postsynaptic_neurons: the number of postsynaptic neurons
        :param initializer: weight initializer function of which argument is zero weight and return is initialized weight
        """
        self.weight = np.zeros([num_presynaptic_neurons, num_postsynaptic_neurons])

        if initializer is None:
            initializer = self.random_initialize
        self.weight = initializer(self.weight)

        assert self.weight.shape == (
            num_presynaptic_neurons,
            num_postsynaptic_neurons,
        ), f"Initialized weight shape {self.weight.shape} is different from ({num_presynaptic_neurons}, {num_postsynaptic_neurons})!"

    def __call__(self, input_potentials: List[float]) -> List[float]:
        """
        Multiply synapse weight about input_potentials and return weighted potentials

        :param input_potentials: input potentials shaped [NumPresynapticNeurons]
        :returns: weighted potentials shaped [NumPostsynapticNeurons]
        """
        assert len(input_potentials) == len(
            self.weight
        ), f"the number of input_potentials {len(input_potentials)} is different from {len(self.weight)}!"

        weighted_potentials = np.matmul(input_potentials, self.weight)
        return weighted_potentials

    @staticmethod
    def random_initialize(weight: np.ndarray) -> np.ndarray:
        return np.random.uniform(0.0, 1.0, weight.shape)
