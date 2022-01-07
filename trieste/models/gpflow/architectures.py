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

""" This file contains builders for some GPflow models supported in Trieste. """

from __future__ import annotations

import math
from typing import Optional

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import GPR, SGPR, SVGP, VGP

from ...data import Dataset
from ...space import Box
from ...types import TensorType


def build_gpr(
    data: Dataset,
    search_space: Box,
    kernel_priors: bool = True,
    noise_free: bool = False,
    likelihood_variance: Optional[float] = None,
) -> GPR:
    """
    Constructs a :class:`~gpflow.models.GPR` model with sensible initial parameters and
    (optionally) priors. We use :class:`~gpflow.kernels.Matern52` kernel in the model as we found
    it to be effective in most settings. We assume inputs are normalised to the unit hypercube for
    setting sensible initial parameters for the kernel.

    We set priors for kernel hyperparameters by default in order to stabilize model fitting. We
    found the priors below to be highly effective for objective functions defined over the unit
    hypercube. For objective functions with different scaling, other priors will likely be more
    appropriate. Using priors allows for using maximum a posteriori estimate of these kernel
    parameters during model fitting.

    :param data: Dataset from the initial design, used for estimating the variance of observations.
    :param noise_free: If set to `True` Gaussian likelihood paramater is set to
        non-trainable. By default set to `False`.
    :param priors: If set to `True` (default) priors are set for kernel parameters.
    :return: A :class:`~gpflow.models.GPR` model.
    """
    empirical_mean, empirical_variance, _ = _get_data_stats(data)

    kernel = _get_kernel(empirical_variance, search_space, kernel_priors)
    mean = gpflow.mean_functions.Constant(empirical_mean)

    if likelihood_variance is None:
        noise_variance = 0.1 * empirical_variance
    else:
        noise_variance = likelihood_variance

    model = gpflow.models.GPR(data.astuple(), kernel, mean, noise_variance)

    gpflow.set_trainable(model.likelihood, (not noise_free))

    return model


def build_sgpr(
    data: Dataset,
    search_space: Box,
    kernel_priors: bool = True,
    noise_free: bool = False,
    likelihood_variance: Optional[float] = None,
    num_inducing_points: Optional[int] = None,
    trainable_inducing_points: bool = False,
) -> SGPR:
    empirical_mean, empirical_variance, _ = _get_data_stats(data)

    kernel = _get_kernel(empirical_variance, search_space, kernel_priors)
    mean = gpflow.mean_functions.Constant(empirical_mean)

    inducing_points = _get_inducing_points(search_space, num_inducing_points)

    if likelihood_variance is None:
        noise_variance = 0.1 * empirical_variance
    else:
        noise_variance = likelihood_variance

    model = gpflow.models.SGPR(
        data.astuple(), kernel, inducing_points, mean_function=mean, noise_variance=noise_variance
    )

    gpflow.set_trainable(model.likelihood, (not noise_free))
    gpflow.set_trainable(model.inducing_variable, trainable_inducing_points)

    return model


def build_vgp(
    data: Dataset,
    search_space: Box,
    classification: bool = False,
    kernel_priors: bool = True,
    noise_free: bool = False,
    likelihood_variance: Optional[float] = None,
) -> VGP:
    empirical_mean, empirical_variance, _ = _get_data_stats(data)

    if classification:
        if noise_free:
            empirical_variance = 100.0
        else:
            empirical_variance = 1.0
        model_likelihood = gpflow.likelihoods.Bernoulli()
    else:
        if likelihood_variance is None:
            noise_variance = 0.1 * empirical_variance
        else:
            noise_variance = likelihood_variance
        model_likelihood = gpflow.likelihoods.Gaussian(noise_variance)

    kernel = _get_kernel(
        empirical_variance, search_space, kernel_priors, classification, noise_free
    )
    mean = gpflow.mean_functions.Constant(empirical_mean)

    model = gpflow.models.VGP(data.astuple(), kernel, model_likelihood, mean_function=mean)

    gpflow.set_trainable(model.likelihood, (not noise_free))
    if classification and noise_free:
        gpflow.set_trainable(model.kernel.variance, False)

    return model


def build_svgp(
    data: Dataset,
    search_space: Box,
    classification: bool = False,
    kernel_priors: bool = True,
    noise_free: bool = False,
    likelihood_variance: Optional[float] = None,
    num_inducing_points: Optional[int] = None,
    trainable_inducing_points: bool = False,
) -> SVGP:
    empirical_mean, empirical_variance, num_data_points = _get_data_stats(data)

    if classification:
        if noise_free:
            empirical_variance = 100.0
        else:
            empirical_variance = 1.0
        model_likelihood = gpflow.likelihoods.Bernoulli()
    else:
        if likelihood_variance is None:
            noise_variance = 0.1 * empirical_variance
        else:
            noise_variance = likelihood_variance
        model_likelihood = gpflow.likelihoods.Gaussian(noise_variance)

    kernel = _get_kernel(
        empirical_variance, search_space, kernel_priors, classification, noise_free
    )
    mean = gpflow.mean_functions.Constant(empirical_mean)

    inducing_points = _get_inducing_points(search_space, num_inducing_points)

    model = SVGP(
        kernel,
        model_likelihood,
        inducing_points,
        mean_function=mean,
        num_data=num_data_points,
    )

    gpflow.set_trainable(model.likelihood, (not noise_free))
    gpflow.set_trainable(model.inducing_variable, trainable_inducing_points)
    if classification:
        gpflow.set_trainable(model.kernel.variance, False)

    return model


def _get_data_stats(data: Dataset) -> tuple[TensorType, TensorType, int]:
    empirical_variance = tf.math.reduce_variance(data.observations)
    empirical_mean = tf.math.reduce_mean(data.observations)
    num_data_points = len(data.observations)

    return empirical_mean, empirical_variance, num_data_points


def _get_kernel(
    variance: TensorType,
    search_space: Box,
    priors: bool,
    classification: bool = False,
    noise_free: bool = False,
) -> gpflow.kernels.Kernel:
    dim = search_space.upper.shape[-1]
    ub, lb = search_space.upper[-1], search_space.lower[-1]

    lengthscales = [0.2 * (ub - lb) * math.sqrt(dim)] * dim
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=lengthscales)

    if priors:
        prior_scale = tf.cast(1.0, dtype=gpflow.default_float())
        if not (classification and noise_free):
            kernel.variance.prior = tfp.distributions.LogNormal(tf.math.log(variance), prior_scale)
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(kernel.lengthscales), prior_scale
        )

    return kernel


def _get_inducing_points(search_space: Box, num_inducing_points: Optional[int]) -> TensorType:
    if num_inducing_points is None:
        num_inducing_points = min(500, 25 * search_space.upper.shape[-1])

    inducing_points = search_space.sample_sobol(num_inducing_points)

    return inducing_points
