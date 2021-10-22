# Copyright 2021 The Trieste Contributors
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

from __future__ import annotations

import copy
from typing import TypeVar, Optional, Union

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset

from gpflow.models import GPR
from gpflow.conditionals import base_conditional
from gpflow.base import default_float, TensorType, Parameter
from gpflow.models.training_mixins import RegressionData
from gpflux.sampling.sample import Sample
from gpflux.sampling.utils import draw_conditional_sample

M = TypeVar("M", bound=tf.Module)
""" A type variable bound to :class:`tf.Module`. """


def module_deepcopy(self: M, memo: dict[int, object]) -> M:
    r"""
    This function provides a workaround for `a bug`_ in TensorFlow Probability (fixed in `version
    0.12`_) where a :class:`tf.Module` cannot be deep-copied if it has
    :class:`tfp.bijectors.Bijector` instances on it. The function can be used to directly copy an
    object ``self`` as e.g. ``module_deepcopy(self, {})``, but it is perhaps more useful as an
    implemention for :meth:`__deepcopy__` on classes, where it can be used as follows:

    .. _a bug: https://github.com/tensorflow/probability/issues/547
    .. _version 0.12: https://github.com/tensorflow/probability/releases/tag/v0.12.1

    .. testsetup:: *

        >>> import tensorflow_probability as tfp

    >>> class Foo(tf.Module):
    ...     example_bijector = tfp.bijectors.Exp()
    ...
    ...     __deepcopy__ = module_deepcopy

    Classes with this method can be deep-copied even if they contain
    :class:`tfp.bijectors.Bijector`\ s.

    :param self: The object to copy.
    :param memo: References to existing deep-copied objects (by object :func:`id`).
    :return: A deep-copy of ``self``.
    """
    gpflow.utilities.reset_cache_bijectors(self)

    new = self.__new__(type(self))
    memo[id(self)] = new

    for name, value in self.__dict__.items():
        setattr(new, name, copy.deepcopy(value, memo))

    return new


def assert_data_is_compatible(new_data: Dataset, existing_data: Dataset) -> None:
    if new_data.query_points.shape[-1] != existing_data.query_points.shape[-1]:
        raise ValueError(
            f"Shape {new_data.query_points.shape} of new query points is incompatible with"
            f" shape {existing_data.query_points.shape} of existing query points. Trailing"
            f" dimensions must match."
        )

    if new_data.observations.shape[-1] != existing_data.observations.shape[-1]:
        raise ValueError(
            f"Shape {new_data.observations.shape} of new observations is incompatible with"
            f" shape {existing_data.observations.shape} of existing observations. Trailing"
            f" dimensions must match."
        )


def randomize_hyperparameters(object: gpflow.Module) -> None:
    """
    Sets hyperparameters to random samples from their constrained domains or (if not constraints
    are available) their prior distributions.

    :param object: Any gpflow Module.
    """
    for param in object.trainable_parameters:
        if isinstance(param.bijector, tfp.bijectors.Sigmoid):
            sample = tf.random.uniform(
                param.bijector.low.shape,
                minval=param.bijector.low,
                maxval=param.bijector.high,
                dtype=param.bijector.low.dtype,
            )
            param.assign(sample)
        elif param.prior is not None:
            param.assign(param.prior.sample())


def squeeze_hyperparameters(
    object: gpflow.Module, alpha: float = 1e-2, epsilon: float = 1e-7
) -> None:
    """
    Squeezes the parameters to be strictly inside their range defined by the Sigmoid,
    or strictly greater than the limit defined by the Shift+Softplus.
    This avoids having Inf unconstrained values when the parameters are exactly at the boundary.

    :param object: Any gpflow Module.
    :param alpha: the proportion of the range with which to squeeze for the Sigmoid case
    :param epsilon: the value with which to offset the shift for the Softplus case.
    :raise ValueError: If ``alpha`` is not in (0,1) or epsilon <= 0
    """

    if not (0 < alpha < 1):
        raise ValueError(f"squeeze factor alpha must be in (0, 1), found {alpha}")

    if not (0 < epsilon):
        raise ValueError(f"offset factor epsilon must be > 0, found {epsilon}")

    for param in object.trainable_parameters:
        if isinstance(param.bijector, tfp.bijectors.Sigmoid):
            delta = (param.bijector.high - param.bijector.low) * alpha
            squeezed_param = tf.math.minimum(param, param.bijector.high - delta)
            squeezed_param = tf.math.maximum(squeezed_param, param.bijector.low + delta)
            param.assign(squeezed_param)
        elif (
            isinstance(param.bijector, tfp.bijectors.Chain)
            and len(param.bijector.bijectors) == 2
            and isinstance(param.bijector.bijectors[0], tfp.bijectors.Shift)
            and isinstance(param.bijector.bijectors[1], tfp.bijectors.Softplus)
        ):
            if isinstance(param.bijector.bijectors[0], tfp.bijectors.Shift) and isinstance(
                param.bijector.bijectors[1], tfp.bijectors.Softplus
            ):
                low = param.bijector.bijectors[0].shift
                squeezed_param = tf.math.maximum(param, low + epsilon * tf.ones_like(param))
                param.assign(squeezed_param)


def sample_gpr(
    data: RegressionData,
    kernel: gpflow.kernels.Kernel,
    noise_variance: Union[float, Parameter],
    mean_function: gpflow.mean_functions.MeanFunction = gpflow.mean_functions.Zero(),
) -> Sample:

    def add_noise_cov(K: tf.Tensor, likelihood_variance: Parameter) -> tf.Tensor:
        """
        Returns K + σ² I, where σ² is the likelihood noise variance (scalar),
        and I is the corresponding identity matrix.
        """
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill(tf.shape(k_diag), likelihood_variance)
        return tf.linalg.set_diag(K, k_diag + s_diag)

    class GPRSampleConditional(Sample):
        X = None  # [N_old, D]
        P = tf.shape(data[1])[-1]
        f = tf.zeros((0, P), dtype=default_float())  # [N_old, P]

        def __call__(self, X_new: TensorType) -> tf.Tensor:
            N_old = tf.shape(self.f)[0]
            N_new = tf.shape(X_new)[0]

            if self.X is None:
                self.X = X_new
            else:
                self.X = tf.concat([self.X, X_new], axis=0)

            err = data[1] - mean_function(data[0])

            kmm = kernel(data[0])
            knn = kernel(self.X, full_cov=True)
            kmn = kernel(data[0], self.X)
            kmm_plus_s = add_noise_cov(kmm, noise_variance)

            mean, cov = base_conditional(
                kmn, kmm_plus_s, knn, err, full_cov=True, white=False
            )  # mean: [N_old+N_new, P], cov: [P, N_old+N_new, N_old+N_new]
            mean = tf.linalg.matrix_transpose(mean)
            f_old = tf.linalg.matrix_transpose(self.f)
            f_new = draw_conditional_sample(mean, cov, f_old)
            f_new = tf.linalg.matrix_transpose(f_new)
            self.f = tf.concat([self.f, f_new], axis=0)

            tf.debugging.assert_equal(tf.shape(self.f), [N_old + N_new, self.P])
            tf.debugging.assert_equal(tf.shape(f_new), [N_new, self.P])

            return f_new

    return GPRSampleConditional()
