from dataclasses import dataclass


@dataclass
class NeuronConfig:
    """Config to initialze Neuron"""

    #: potential when resting
    rest_potential: float

    #: minimum potential
    min_potential: float

    #: threshold that fire a spike when potential is over than
    threshold_potential: float

    #: added potential when excited
    excitatory_potential: float

    #: decreasing potential by leakage_factor
    leakage_factor: float

    #: the number of time step in which neuron can't be receive input potential
    rest_period: int
