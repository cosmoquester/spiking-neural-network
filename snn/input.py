from typing import Iterator, List, Union

import numpy as np

from .units import ReceptiveField


class RandomStimulator:
    def __init__(self, shape=(), timestep: Union[int, float] = float("inf")):
        self.shape = shape
        self.timestep = timestep

    def __iter__(self) -> Iterator[np.ndarray]:
        """Return random value generator whose length is timestep."""
        index = 0

        while index < self.timestep:
            yield np.random.randint(0, 2, self.shape)
            index += 1


class ImageEncoder:
    """Make spike train from 2D binary image with ReceptiveField"""

    def __init__(
        self,
        receptive_field: ReceptiveField,
        timestep: int,
        freq_multiplier: float,
        freq_bias: float,
        image: List[List[float]] = None,
    ):
        """
        Initialize ImageEncoder

        :param receptive_field: ReceptiveField for read image
        :param timestep: total number of timesteps for encoding
        :param freq_multiplier

        """
        self.receptive_field = receptive_field
        self.timestep = timestep
        self.freq_multiplier = freq_multiplier
        self.freq_bias = freq_bias

        # Spike Train shaped [TimeStep, NumRows, NumColumns]
        self._spike_train = None
        self._cur_timestep = 0

        if image is not None:
            self.set_spike_train(image)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Return iterator containing spike train of all timestep"""
        assert self._spike_train is not None, '"set_spike_train" method have not be called!"'

        for spike in self._spike_train:
            yield spike

    def set_spike_train(self, image: List[List[float]]) -> None:
        """
        Set spike train using a input image

        :param image: 2D bianry array of a image
        """
        potential = self.receptive_field(image)
        self._spike_train = self.rate_encoding(potential)

    def rate_encoding(self, potential: List[List[float]]) -> np.ndarray:
        """
        Encode input potentials as scaled frequency

        :param potential: input potential to encode
        :returns: spike train shaped [TimeStep, NumRows, NumColumns]
        """
        potential_array = np.array(potential, np.float32)
        frequency = np.ceil(potential_array * self.freq_multiplier + self.freq_bias)
        frequency_2d = frequency[:, :, np.newaxis].repeat(self.timestep, axis=2)

        for frequency_1d in frequency_2d:
            for f_time in frequency_1d:
                f_time[:: int(f_time[0])] = np.nan
        spike_train = np.isnan(frequency_2d).astype(np.int32).transpose((2, 0, 1))
        return spike_train
