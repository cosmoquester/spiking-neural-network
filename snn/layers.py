from typing import Callable, List, Optional, Tuple

import numpy as np

from .configs import NeuronConfig
from .units import Neurons, Synapse


class FeedForward:
    """FeedForward layer consisting of Neurons and Synapse"""

    def __init__(
        self,
        cfg: NeuronConfig,
        num_presynaptic_neurons: int,
        num_postsynaptic_neurons: int,
        synapse_initializer: Optional[Callable] = None,
    ):
        """
        Initialize FeedForward layer

        :param cfg: neuron's config
        :param num_presynaptic_neurons: the number of presynaptic neurons
        :param num_postsynaptic_neurons: the number of postsynaptic neurons
        :param synapse_initializer: initializer for syanpse
        """
        self.neurons = Neurons(num_presynaptic_neurons, cfg)
        self.synapse = Synapse(num_presynaptic_neurons, num_postsynaptic_neurons, synapse_initializer)

    def __call__(self, input_potentials: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Feed input potentials to neurons and synapse

        :param input_potentials: input potentials of each neuron
        :returns: tuple of fire spikes, neurons potential, synapse weighted output potential
        """
        fire_spikes, neuron_potentials = self.neurons(input_potentials)
        weighted_potential = self.synapse(fire_spikes)
        return fire_spikes, neuron_potentials, weighted_potential
