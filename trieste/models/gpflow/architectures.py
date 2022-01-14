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

""" This file contains builders for GPflow models supported in Trieste. """

from __future__ import annotations

import math
from typing import Optional

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import GPR, SGPR, SVGP, VGP

from ...data import Dataset
from ...space import Box, SearchSpace
from ...types import TensorType

KERNEL_LENGTHSCALE = tf.cast(0.2, dtype=gpflow.default_float())
"""
Default value of the kernel lengthscale parameter.
"""


KERNEL_PRIOR_SCALE = tf.cast(1.0, dtype=gpflow.default_float())
"""
Default value of the scaling factor for the kernel lengthscale and variance parameters.
"""


CLASSIFICATION_KERNEL_VARIANCE_NOISE_FREE = tf.cast(100.0, dtype=gpflow.default_float())
"""
Default value of the kernel variance parameter for classification models in the noise free case.
"""


CLASSIFICATION_KERNEL_VARIANCE = tf.cast(1.0, dtype=gpflow.default_float())
"""
Default value of the kernel variance parameter for classification models.
"""


MAX_NUM_INDUCING_POINTS = tf.cast(500, dtype=tf.int32)
"""
Default maximum number of inducing points.
"""


NUM_INDUCING_POINTS_PER_DIM = tf.cast(25, dtype=tf.int32)
"""
Default number of inducing points per dimension of the search space.
"""


SNR_LIKELIHOOD = tf.cast(10, dtype=gpflow.default_float())
"""
Default value used for initializing (noise) variance parameter of the likelihood function.
If user does not specify it, the noise variance is set to maintain the signal to noise ratio
determined by this default value. Signal variance in the kernel is set to the empirical variance.
"""


def build_gpr(
    data: Dataset,
    search_space: SearchSpace,
    kernel_priors: bool = True,
    likelihood_variance: Optional[float] = None,
    trainable_likelihood: bool = False,
) -> GPR:
    """
    Build a :class:`~gpflow.models.GPR` model with sensible initial parameters and
    priors. We use :class:`~gpflow.kernels.Matern52` kernel and
    :class:`~gpflow.mean_functions.Constant` mean function in the model as we found this combination
    to be effective in most settings.

    We set priors for kernel hyperparameters by default in order to stabilize model fitting. We
    found the priors below to be highly effective for objective functions defined over the unit
    hypercube. They do seem to work for other search space sizes, but we advise caution when using
    them in such search spaces. Using priors allows for using maximum a posteriori estimate of
    these kernel parameters during model fitting.

    Note that although we scale parameters as a function of the size of the search space, ideally
    inputs should be normalised to the unit hypercube before building a model.

    :param data: Dataset from the initial design, used for estimating the variance of observations.
    :param search_space: Search space for performing Bayesian optimization, used for scaling the
        parameters.
    :param kernel_priors: If set to `True` (default) priors are set for kernel parameters (variance
        and lengthscale).
    :param likelihood_variance: Likelihood (noise) variance parameter can be optionally set to a
        certain value. If left unspecified (default), the noise variance is set to maintain the
        signal to noise ratio of value given by ``SNR_LIKELIHOOD``, where signal variance in the
        kernel is set to the empirical variance.
    :param trainable_likelihood: If set to `True` Gaussian likelihood parameter is set to
        non-trainable. By default set to `False`.
    :return: A :class:`~gpflow.models.GPR` model.
    """
    empirical_mean, empirical_variance, _ = _get_data_stats(data)

    kernel = _get_kernel(empirical_variance, search_space, kernel_priors, kernel_priors)
    mean = gpflow.mean_functions.Constant(empirical_mean)

    if likelihood_variance is None:
        noise_variance = empirical_variance / SNR_LIKELIHOOD ** 2
    else:
        noise_variance = tf.cast(likelihood_variance, dtype=gpflow.default_float())

    model = gpflow.models.GPR(data.astuple(), kernel, mean, noise_variance)

    gpflow.set_trainable(model.likelihood, trainable_likelihood)

    return model


def build_sgpr(
    data: Dataset,
    search_space: SearchSpace,
    kernel_priors: bool = True,
    likelihood_variance: Optional[float] = None,
    trainable_likelihood: bool = False,
    num_inducing_points: Optional[int] = None,
    trainable_inducing_points: bool = False,
) -> SGPR:
    """
    Build a :class:`~gpflow.models.SGPR` model with sensible initial parameters and
    priors. We use :class:`~gpflow.kernels.Matern52` kernel and
    :class:`~gpflow.mean_functions.Constant` mean function in the model as we found this combination
    to be effective in most settings.

    We set priors for kernel hyperparameters by default in order to stabilize model fitting. We
    found the priors below to be highly effective for objective functions defined over the unit
    hypercube. They do seem to work for other search space sizes, but we advise caution when using
    them in such search spaces. Using priors allows for using maximum a posteriori estimate of
    these kernel parameters during model fitting.

    For performance reasons number of inducing points should not be changed during Bayesian
    optimization. Hence, even if the initial dataset is smaller, we advise setting this to a higher
    number. By default inducing points are set to Sobol samples for the continuous search space,
    and simple random samples for discrete or mixed search spaces. This carries
    the risk that optimization gets stuck if they are not trainable, which calls for adaptive
    inducing point selection during the optimization. This functionality will be added to Trieste
    in future.

    Note that although we scale parameters as a function of the size of the search space, ideally
    inputs should be normalised to the unit hypercube before building a model.

    :param data: Dataset from the initial design, used for estimating the variance of observations.
    :param search_space: Search space for performing Bayesian optimization, used for scaling the
        parameters.
    :param kernel_priors: If set to `True` (default) priors are set for kernel parameters (variance
        and lengthscale).
    :param likelihood_variance: Likelihood (noise) variance parameter can be optionally set to a
        certain value. If left unspecified (default), the noise variance is set to maintain the
        signal to noise ratio of value given by ``SNR_LIKELIHOOD``, where signal variance in the
        kernel is set to the empirical variance.
    :param trainable_likelihood: If set to `True` Gaussian likelihood parameter is set to
        be trainable. By default set to `False`.
    :param num_inducing_points: The number of inducing points can be optionally set to a
        certain value. If left unspecified (default), this number is set to either
        ``NUM_INDUCING_POINTS_PER_DIM``*dimensionality of the search space or value given by
        ``MAX_NUM_INDUCING_POINTS``, whichever is smaller.
    :param trainable_inducing_points: If set to `True` inducing points will be set to
        be trainable. This option should be used with caution. By default set to `False`.
    :return: An :class:`~gpflow.models.SGPR` model.
    """
    empirical_mean, empirical_variance, _ = _get_data_stats(data)

    kernel = _get_kernel(empirical_variance, search_space, kernel_priors, kernel_priors)
    mean = gpflow.mean_functions.Constant(empirical_mean)

    inducing_points = _get_inducing_points(search_space, num_inducing_points)

    if likelihood_variance is None:
        noise_variance = empirical_variance / SNR_LIKELIHOOD ** 2
    else:
        noise_variance = tf.cast(likelihood_variance, dtype=gpflow.default_float())

    model = SGPR(
        data.astuple(), kernel, inducing_points, mean_function=mean, noise_variance=noise_variance
    )

    gpflow.set_trainable(model.likelihood, trainable_likelihood)
    gpflow.set_trainable(model.inducing_variable, trainable_inducing_points)

    return model


def build_vgp(
    data: Dataset,
    search_space: SearchSpace,
    kernel_priors: bool = True,
    noise_free: bool = False,
    kernel_variance: Optional[float] = None,
) -> VGP:
    """
    Build a :class:`~gpflow.models.VGP` binary classification model with sensible initial
    parameters and priors. We use :class:`~gpflow.kernels.Matern52` kernel and
    :class:`~gpflow.mean_functions.Constant` mean function in the model as we found this combination
    to be effective in most settings.

    We set priors for kernel hyperparameters by default in order to stabilize model fitting. We
    found the priors below to be highly effective for objective functions defined over the unit
    hypercube. They do seem to work for other search space sizes, but we advise caution when using
    them in such search spaces. Using priors allows for using maximum a posteriori estimate of
    these kernel parameters during model fitting. In the ``noise_free`` case we do not use prior
    for the kernel variance parameters.

    Note that although we scale parameters as a function of the size of the search space, ideally
    inputs should be normalised to the unit hypercube before building a model.

    :param data: Dataset from the initial design, used for estimating the variance of observations.
    :param search_space: Search space for performing Bayesian optimization, used for scaling the
        parameters.
    :param kernel_priors: If set to `True` (default) priors are set for kernel parameters (variance
        and lengthscale). In the ``noise_free`` case kernel variance prior is not set.
    :param noise_free: If  there is a prior information that the classification problem is a
        deterministic one, this should be set to `True` and kernel variance will be fixed to a
        higher default value ``CLASSIFICATION_KERNEL_VARIANCE_NOISE_FREE`` leading to sharper
        classification boundary. In this case prior for the kernel variance parameter is also not
        set. By default set to `False`.
    :param kernel_variance: Kernel variance parameter can be optionally set to a
        certain value. If left unspecified (default), the kernel variance is set to
        ``CLASSIFICATION_KERNEL_VARIANCE_NOISE_FREE`` in the ``noise_free`` case and to
        ``CLASSIFICATION_KERNEL_VARIANCE`` otherwise.
    :return: A :class:`~gpflow.models.VGP` model.
    """
    if kernel_variance is not None:
        variance = tf.cast(kernel_variance, dtype=gpflow.default_float())
    else:
        if noise_free:
            variance = CLASSIFICATION_KERNEL_VARIANCE_NOISE_FREE
        else:
            variance = CLASSIFICATION_KERNEL_VARIANCE

    if noise_free:
        add_prior_to_variance = False
    else:
        add_prior_to_variance = kernel_priors

    mean_constant = tf.cast(0.0, dtype=gpflow.default_float())
    model_likelihood = gpflow.likelihoods.Bernoulli()
    kernel = _get_kernel(variance, search_space, kernel_priors, add_prior_to_variance)
    mean = gpflow.mean_functions.Constant(mean_constant)

    model = VGP(data.astuple(), kernel, model_likelihood, mean_function=mean)

    gpflow.set_trainable(model.kernel.variance, (not noise_free))

    return model


def build_svgp(
    data: Dataset,
    search_space: SearchSpace,
    classification: bool = False,
    kernel_priors: bool = True,
    likelihood_variance: Optional[float] = None,
    trainable_likelihood: bool = False,
    num_inducing_points: Optional[int] = None,
    trainable_inducing_points: bool = False,
) -> SVGP:
    """
    Build a :class:`~gpflow.models.SVGP` model with sensible initial parameters and
    priors. We use :class:`~gpflow.kernels.Matern52` kernel and
    :class:`~gpflow.mean_functions.Constant` mean function in the model as we found this combination
    to be effective in most settings. Both regression and binary classification models are
    available.

    We set priors for kernel hyperparameters by default in order to stabilize model fitting. We
    found the priors below to be highly effective for objective functions defined over the unit
    hypercube. They do seem to work for other search space sizes, but we advise caution when using
    them in such search spaces. Using priors allows for using maximum a posteriori estimate of
    these kernel parameters during model fitting.

    For performance reasons number of inducing points should not be changed during Bayesian
    optimization. Hence, even if the initial dataset is smaller, we advise setting this to a higher
    number. By default inducing points are set to Sobol samples for the continuous search space,
    and simple random samples for discrete or mixed search spaces. This carries
    the risk that optimization gets stuck if they are not trainable, which calls for adaptive
    inducing point selection during the optimization. This functionality will be added to Trieste
    in future.

    Note that although we scale parameters as a function of the size of the search space, ideally
    inputs should be normalised to the unit hypercube before building a model.

    :param data: Dataset from the initial design, used for estimating the variance of observations.
    :param search_space: Search space for performing Bayesian optimization, used for scaling the
        parameters.
    :param classification: If a classification model is needed, this should be set to `True`, in
        which case a Bernoulli likelihood will be used. If a regression model is required, this
        should be set to `False` (default), in which case a Gaussian likelihood is used.
    :param kernel_priors: If set to `True` (default) priors are set for kernel parameters (variance
        and lengthscale).
    :param likelihood_variance: Likelihood (noise) variance parameter can be optionally set to a
        certain value. If left unspecified (default), the noise variance is set to maintain the
        signal to noise ratio of value given by ``SNR_LIKELIHOOD``, where signal variance in the
        kernel is set to the empirical variance. This argument is ignored in the classification
        case.
    :param trainable_likelihood: If set to `True` likelihood parameter is set to
        be trainable. By default set to `False`. This argument is ignored in the classification
        case.
    :param num_inducing_points: The number of inducing points can be optionally set to a
        certain value. If left unspecified (default), this number is set to either
        ``NUM_INDUCING_POINTS_PER_DIM``*dimensionality of the search space or value given by
        ``MAX_NUM_INDUCING_POINTS``, whichever is smaller.
    :param trainable_inducing_points: If set to `True` inducing points will be set to
        be trainable. This option should be used with caution. By default set to `False`.
    :return: An :class:`~gpflow.models.SVGP` model.
    """
    empirical_mean, empirical_variance, num_data_points = _get_data_stats(data)

    if classification:
        empirical_variance = CLASSIFICATION_KERNEL_VARIANCE
        empirical_mean = tf.cast(0.0, dtype=gpflow.default_float())
        model_likelihood = gpflow.likelihoods.Bernoulli()
    else:
        if likelihood_variance is None:
            noise_variance = empirical_variance / SNR_LIKELIHOOD ** 2
        else:
            noise_variance = tf.cast(likelihood_variance, dtype=gpflow.default_float())
        model_likelihood = gpflow.likelihoods.Gaussian(noise_variance)

    kernel = _get_kernel(empirical_variance, search_space, kernel_priors, kernel_priors)
    mean = gpflow.mean_functions.Constant(empirical_mean)

    inducing_points = _get_inducing_points(search_space, num_inducing_points)

    model = SVGP(
        kernel,
        model_likelihood,
        inducing_points,
        mean_function=mean,
        num_data=num_data_points,
    )

    if not classification:
        gpflow.set_trainable(model.likelihood, trainable_likelihood)
    gpflow.set_trainable(model.inducing_variable, trainable_inducing_points)

    return model


def _get_data_stats(data: Dataset) -> tuple[TensorType, TensorType, int]:
    empirical_variance = tf.math.reduce_variance(data.observations)
    empirical_mean = tf.math.reduce_mean(data.observations)
    num_data_points = len(data.observations)

    return empirical_mean, empirical_variance, num_data_points


def _get_kernel(
    variance: TensorType,
    search_space: SearchSpace,
    add_prior_to_lengthscale: bool,
    add_prior_to_variance: bool,
) -> gpflow.kernels.Kernel:

    lengthscales = (
        KERNEL_LENGTHSCALE
        * (search_space.upper - search_space.lower)
        * math.sqrt(search_space.dimension)
    )
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=lengthscales)

    if add_prior_to_lengthscale:
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(kernel.lengthscales), KERNEL_PRIOR_SCALE
        )
    if add_prior_to_variance:
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.math.log(variance), KERNEL_PRIOR_SCALE
        )

    return kernel


def _get_inducing_points(
    search_space: SearchSpace, num_inducing_points: Optional[int]
) -> TensorType:
    if num_inducing_points is None:
        num_inducing_points = min(
            MAX_NUM_INDUCING_POINTS, NUM_INDUCING_POINTS_PER_DIM * search_space.dimension
        )
    if isinstance(search_space, Box):
        inducing_points = search_space.sample_sobol(num_inducing_points)
    else:
        inducing_points = search_space.sample(num_inducing_points)

    return inducing_points
