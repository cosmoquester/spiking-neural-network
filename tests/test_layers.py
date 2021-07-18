import numpy as np

from snn.configs import NeuronConfig
from snn.input import RandomStimulator
from snn.layers import FeedForward

cfg = NeuronConfig(
    rest_potential=0.0,
    min_potential=-1.0,
    threshold_potential=25.0,
    excitatory_potential=4.0,
    leakage_factor=0.25,
    rest_period=40,
)


def test_feed_foward():
    num_neurons = 5
    num_post_neurons = 3
    feedforward = FeedForward(cfg, num_neurons, num_post_neurons)
    spikes, neuron_potentials, output_potentials = feedforward(next(iter(RandomStimulator([num_neurons]))))

    np.testing.assert_equal(len(spikes), num_neurons)
    np.testing.assert_equal(len(neuron_potentials), num_neurons)
    np.testing.assert_equal(output_potentials.shape, [num_post_neurons])
