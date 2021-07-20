import numpy as np
import pytest

from snn.input import ImageEncoder, RandomStimulator
from snn.units import ReceptiveField


def test_random_stimulator():
    shape = [3, 4]
    timestep = 10
    stimulator = RandomStimulator(shape=shape, timestep=timestep)
    outputs = list(stimulator)

    assert len(outputs) == len(stimulator) == timestep
    np.testing.assert_equal(outputs[0].shape, shape)


def test_image_stimulator():
    timestep = 100
    receptive_field = ReceptiveField([3, 3])
    stimulator = ImageEncoder(receptive_field, timestep, 0.102, 52.02)

    with pytest.raises(AssertionError):
        next(iter(stimulator))

    random_image = np.random.randn(32, 32)
    stimulator.set_spike_train(random_image)
    assert len(stimulator) == timestep

    outputs = list(stimulator)
    assert len(outputs) == timestep
    assert np.all(np.logical_or(outputs[0] == 0, outputs[0] == 1))
    np.testing.assert_equal(outputs[0].shape, [32 * 32])
    np.testing.assert_equal(outputs[0], stimulator[0])
    np.testing.assert_equal(outputs[0], stimulator.get(0))

    stimulator = ImageEncoder(ReceptiveField([1, 1], False), 10, 1.0, 0.0)
    spike_train = stimulator.rate_encoding([[3]]).squeeze()
    np.testing.assert_equal(spike_train, [1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
