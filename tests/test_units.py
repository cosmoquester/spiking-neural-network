from snn.configs import NeuronConfig
from snn.input import RandomStimulator
from snn.units import Neuron, Neurons

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
