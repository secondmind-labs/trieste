# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module is the home of the sampling functionality required by Trieste's models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import tensorflow as tf

from ..types import TensorType


class ModelSampler(ABC):
    r"""
    A :class:`ModelSampler` samples a specific quantity across a discrete set of points
    according to an underlying :class:`ProbabilisticModel`.
    """

    def __init__(self, sample_size: int):
        """
        :param sample_size: The desired number of samples.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)

        self._sample_size = sample_size

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._sample_size!r})"

    @abstractmethod
    def sample(self, at: TensorType) -> TensorType:
        """
        :param at: Input points that define the sampler.
        :return: Samples.
        """
