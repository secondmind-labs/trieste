from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, overload

import tensorflow as tf
import tensorflow_probability as tfp

from ..type import TensorType


class Design(ABC):
    """
    A :class:`Design` base class for implementing custom sampling/Design of Experiment methods.
    """

    @abstractmethod
    def generate(
        self,
        num_samples: int,
        domain_dim: int,
        lower: Sequence[float] | TensorType,
        upper: Sequence[float] | TensorType,
    ) -> TensorType:
        """
        :param num_samples: The number of points to sample from this search space.
        :param domain_dim: Dimension of the domain.
        :param lower: Lower bound list
        :param upper: Upper bound list
        :return: ``num_samples`` i.i.d. random points, sampled uniformly from this search space.
        """


class Random(Design):
    @overload
    def generate(
        self,
        num_samples: int,
        domain_dim: int,
        lower: Sequence[float],
        upper: Sequence[float],
        dtype=tf.float64,
    ) -> TensorType:
        ...

    @overload
    def generate(
        self,
        num_samples: int,
        domain_dim: int,
        lower: TensorType,
        upper: TensorType,
        dtype=tf.float64,
    ) -> TensorType:
        ...

    def generate(
        self,
        num_samples: int,
        domain_dim: int,
        lower: Sequence[float] | TensorType,
        upper: Sequence[float] | TensorType,
        dtype=tf.float64,
    ) -> TensorType:
        return tf.random.uniform((num_samples, domain_dim), minval=lower, maxval=upper, dtype=dtype)


class HaltonSequence(Design):
    @overload
    def generate(
        self,
        num_samples: int,
        domain_dim: int,
        lower: Sequence[float],
        upper: Sequence[float],
        dtype=tf.float64,
        seed: int = None,
    ) -> TensorType:
        ...

    @overload
    def generate(
        self,
        num_samples: int,
        domain_dim: int,
        lower: TensorType,
        upper: TensorType,
        dtype=tf.float64,
        seed: int = None,
    ) -> TensorType:
        ...

    def generate(
        self,
        num_samples: int,
        domain_dim: int,
        lower: Sequence[float] | TensorType,
        upper: Sequence[float] | TensorType,
        dtype=tf.float64,
        seed: int = None,
    ) -> TensorType:
        return (upper - lower) * tfp.mcmc.sample_halton_sequence(
            dim=domain_dim, num_results=num_samples, dtype=dtype, seed=seed
        ) + lower


class SobolSequence(Design):
    @overload
    def generate(
        self,
        num_samples: int,
        domain_dim: int,
        lower: Sequence[float],
        upper: Sequence[float],
        dtype=tf.float64,
        skip: int = None,
    ) -> TensorType:
        ...

    @overload
    def generate(
        self,
        num_samples: int,
        domain_dim: int,
        lower: TensorType,
        upper: TensorType,
        dtype=tf.float64,
        skip: int = None,
    ) -> TensorType:
        ...

    def generate(
        self,
        num_samples: int,
        domain_dim: int,
        lower: Sequence[float] | TensorType,
        upper: Sequence[float] | TensorType,
        dtype=tf.float64,
        skip: int = 0,
    ) -> TensorType:
        return (upper - lower) * tf.math.sobol_sample(
            dim=domain_dim, num_results=num_samples, dtype=dtype, skip=skip
        ) + lower
