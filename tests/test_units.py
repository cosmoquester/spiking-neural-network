import numpy as np

from snn.configs import NeuronConfig
from snn.input import RandomStimulator
from snn.units import Neuron, Neurons, Synapse

cfg = NeuronConfig(
    rest_potential=0.0,
    min_potential=-1.0,
    threshold_potential=25.0,
    excitatory_potential=4.0,
    leakage_factor=0.25,
    rest_period=40,
)


def test_neuron():
    neuron = Neuron(cfg)
    stimulator = RandomStimulator()

    total_time_step = 500
    input_potentials = []
    neuron_potentials = []
    output_spikes = []

    for _ in range(total_time_step):
        input_potential = stimulator()
        spike, neuron_potential = neuron(input_potential)

        input_potentials.append(input_potential)
        neuron_potentials.append(neuron_potential)
        output_spikes.append(int(spike))

    assert total_time_step >= sum(output_spikes) > 0
    assert max(output_spikes) == 1
    assert min(output_spikes) == 0
    assert max(neuron_potentials) < cfg.threshold_potential + cfg.excitatory_potential + 1
    assert min(neuron_potentials) >= cfg.min_potential


def test_neurons():
    num_neurons = 3
    neurons = Neurons(num_neurons, cfg)
    stimulator = RandomStimulator([num_neurons])

    fire_spikes, output_potentials = neurons(stimulator())
    assert len(fire_spikes) == len(output_potentials) == num_neurons


def test_synapse():
    num_presynaptic_neurons = 2
    num_postsynaptic_neurons = 7
    synapse = Synapse(num_presynaptic_neurons, num_postsynaptic_neurons, initializer=np.zeros_like)
    potentials = [0.3, -0.3]
    weighted_potentials = synapse(potentials)
    np.testing.assert_equal(np.zeros_like(weighted_potentials), weighted_potentials)

    num_presynaptic_neurons = 3
    num_postsynaptic_neurons = 5
    synapse = Synapse(num_presynaptic_neurons, num_postsynaptic_neurons, initializer=np.ones_like)
    potentials = [1.0, 2.0, 3.0]
    weighted_potentials = synapse(potentials)
    np.testing.assert_equal([sum(potentials)] * num_postsynaptic_neurons, weighted_potentials)

    num_presynaptic_neurons = 4
    num_postsynaptic_neurons = 4
    synapse = Synapse(num_presynaptic_neurons, num_postsynaptic_neurons, initializer=lambda w: np.eye(*w.shape))
    potentials = [1.0, 2.0, 3.0, 4.0]
    weighted_potentials = synapse(potentials)
    np.testing.assert_equal(potentials, weighted_potentials)

    num_presynaptic_neurons = 5
    num_postsynaptic_neurons = 2
    synapse = Synapse(num_presynaptic_neurons, num_postsynaptic_neurons)
    potentials = [1.0, 2.0, 3.0, 4.0, 5.0]
    weighted_potentials = synapse(potentials)
    assert np.all(weighted_potentials[0] != weighted_potentials[1])
