#
# Copyright (c) 2021 The GPflux Contributors.
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
#
""" Shared functionality for stationary kernel basis functions. """

from abc import ABC, abstractmethod
from typing import Mapping

import gpflow
import tensorflow as tf
from gpflow.base import TensorType
from gpflux.types import ShapeType


class MultiOutputFourierFeaturesBase(ABC, tf.keras.layers.Layer):
    def __init__(
        self, kernel: gpflow.kernels.MultioutputKernel, n_components: int, **kwargs: Mapping
    ):
        """
        :param kernel: kernel to approximate using a set of Fourier bases.
        Expects a Multioutput Kernel
        :param n_components: number of components (e.g. Monte Carlo samples,
            quadrature nodes, etc.) used to numerically approximate the kernel.
        """
        super(MultiOutputFourierFeaturesBase, self).__init__(**kwargs)
        self.kernel = kernel
        self.n_components = n_components
        if kwargs.get("input_dim", None):
            self._input_dim = kwargs["input_dim"]
            self.build(tf.TensorShape([self._input_dim]))
        else:
            self._input_dim = None

    def call(self, inputs: TensorType) -> tf.Tensor:
        """
        Evaluate the basis functions at ``inputs``.

        :param inputs: The evaluation points, a tensor with the shape ``[N, D]``.

        :return: A tensor with the shape ``[P, N, M]``.mypy
        """

        P = self.kernel.num_latent_gps
        D = tf.shape(inputs)[-1]

        if isinstance(self.kernel, gpflow.kernels.SeparateIndependent):

            for kernel in self.kernel.kernels:
                print(kernel.lengthscales.unconstrained_variable.value())

            sth = [
                kernel.lengthscales[None, None, ...]
                if tf.rank(kernel.lengthscales.unconstrained_variable.value()) == 1
                else kernel.lengthscales[None, None, None, ...]
                for kernel in self.kernel.kernels
            ]

            print("----- printing construction")
            for _ in sth:
                print(_)

            _lengthscales = tf.concat(
                sth,
                axis=0,
            )  # [P, 1, D]
            print("size -f _lengthscales")
            print(_lengthscales)
            tf.debugging.assert_equal(tf.shape(_lengthscales), [P, 1, D])

        elif isinstance(self.kernel, gpflow.kernels.SharedIndependent):
            _lengthscales = tf.tile(
                self.kernel.kernel.lengthscales[None, None, ...]
                if tf.rank(self.kernel.kernel.lengthscales.unconstrained_variable.value()) == 1
                else self.kernel.kernel.lengthscales[None, None, None, ...],
                [P, 1, 1],
            )  # [P, 1, D]
            tf.debugging.assert_equal(tf.shape(_lengthscales), [P, 1, D])
        else:
            raise ValueError("kernel is not supported.")

        X = tf.divide(
            inputs,
            _lengthscales,  # [N, D] or [P, M, D]  # [P, 1, D]
        )  # [P, N, D] or [P, M, D]
        const = self._compute_constant()[..., None, None]  # [P,1,1]
        bases = self._compute_bases(X)  # [P, N, L] for X*, or [P,M,L] in the case of Z
        output = const * bases  # [P, N, L] for X*, or [P,M,L] in the case of Z

        tf.ensure_shape(output, self.compute_output_shape(X.shape))
        return output

    def compute_output_shape(self, input_shape: ShapeType) -> tf.TensorShape:
        """
        Computes the output shape of the layer.
        See `tf.keras.layers.Layer.compute_output_shape()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#compute_output_shape>`_.
        """
        # TODO: Keras docs say "If the layer has not been built, this method
        # will call `build` on the layer." -- do we need to do so?

        tensor_shape = tf.TensorShape(input_shape).with_rank(3)
        output_dim = self._compute_output_dim(input_shape)
        return tensor_shape[:-1].concatenate(output_dim)

    def get_config(self) -> Mapping:
        """
        Returns the config of the layer.
        See `tf.keras.layers.Layer.get_config()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_config>`_.
        """
        config = super(MultiOutputFourierFeaturesBase, self).get_config()
        config.update(
            {
                "kernel": self.kernel,
                "n_components": self.n_components,
                "input_dim": self._input_dim,
            }
        )

        return config

    @abstractmethod
    def _compute_output_dim(self, input_shape: ShapeType) -> int:
        pass

    @abstractmethod
    def _compute_constant(self) -> tf.Tensor:
        """
        Compute normalizing constant for basis functions.
        """
        pass

    @abstractmethod
    def _compute_bases(self, inputs: TensorType) -> tf.Tensor:
        """
        Compute basis functions.
        """
        pass
