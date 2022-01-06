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

from typing import Optional

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import GPR, SGPR, SVGP, VGP

from ...data import Dataset
from ...space import Box


def build_gpr(data: Dataset, trainable_likelihood: bool = False, kernel_priors: bool = True) -> GPR:
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
    :param trainable_likelihood: If set to `True` likelihood paramater is set to
        trainable. By default set to `False`.
    :param priors: If set to `True` (default) priors are set for kernel parameters.
    :return: A :class:`~gpflow.models.GPR` model.
    """
    empirical_variance = tf.math.reduce_variance(data.observations)
    dim = data.query_points.shape[-1]

    kernel = gpflow.kernels.Matern52(variance=empirical_variance, lengthscales=[0.2] * dim)

    if kernel_priors:
        prior_scale = tf.cast(1.0, dtype=gpflow.default_float())
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.math.log(empirical_variance), prior_scale
        )
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(kernel.lengthscales), prior_scale
        )

    model = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)

    if not trainable_likelihood:
        gpflow.set_trainable(model.likelihood, False)

    return model


def build_sgpr(
    data: Dataset,
    search_space: Optional[Box] = None,
    trainable_likelihood: bool = False,
    trainable_inducing_points: bool = False,
) -> SGPR:
    empirical_variance = tf.math.reduce_variance(data.observations)
    dim = data.query_points.shape[-1]
    num_data = len(data.observations)

    kernel = gpflow.kernels.Matern52(variance=empirical_variance, lengthscales=[0.2] * dim)

    if search_space:
        inducing_points = search_space.sample_sobol(num_data)
    else:
        inducing_points = data.query_points[:num_data, :]

    model = gpflow.models.SGPR(data.astuple(), kernel, inducing_points, noise_variance=1e-5)

    if not trainable_likelihood:
        gpflow.set_trainable(model.likelihood, False)

    if not trainable_inducing_points:
        gpflow.set_trainable(model.inducing_variable, False)

    return model


def build_vgp(data: Dataset, trainable_likelihood: bool = False) -> VGP:
    empirical_variance = tf.math.reduce_variance(data.observations)
    dim = data.query_points.shape[-1]

    kernel = gpflow.kernels.Matern52(variance=empirical_variance, lengthscales=[0.2] * dim)

    likelihood = gpflow.likelihoods.Gaussian(variance=1e-5)
    model = gpflow.models.VGP(data.astuple(), kernel, likelihood)

    if not trainable_likelihood:
        gpflow.set_trainable(model.likelihood, False)

    return model


def build_vgp_classification(data: Dataset, trainable_likelihood: bool = False) -> VGP:
    empirical_variance = tf.math.reduce_variance(data.observations)
    dim = data.query_points.shape[-1]

    kernel = gpflow.kernels.Matern52(variance=empirical_variance, lengthscales=[0.2] * dim)

    likelihood = gpflow.likelihoods.Bernoulli()
    model = gpflow.models.VGP(data.astuple(), kernel, likelihood)

    if not trainable_likelihood:
        gpflow.set_trainable(model.likelihood, False)

    return model


def build_svgp(
    data: Dataset,
    search_space: Optional[Box] = None,
    trainable_likelihood: bool = False,
    trainable_inducing_points: bool = False,
) -> SVGP:
    empirical_variance = tf.math.reduce_variance(data.observations)
    dim = data.query_points.shape[-1]
    num_data = len(data.observations)

    kernel = gpflow.kernels.Matern52(variance=empirical_variance, lengthscales=[0.2] * dim)

    if search_space:
        inducing_points = search_space.sample_sobol(num_data)
    else:
        inducing_points = data.query_points[:num_data, :]

    model = SVGP(
        kernel,
        gpflow.likelihoods.Gaussian(variance=1e-5),
        inducing_points,
        num_data=num_data,
    )

    if not trainable_likelihood:
        gpflow.set_trainable(model.likelihood, False)

    if not trainable_inducing_points:
        gpflow.set_trainable(model.inducing_variable, False)

    return model
