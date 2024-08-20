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

"""
This module contains builders for GPflow models supported in Trieste. We found the default
configurations used here to work well in most situation, but they should not be taken as
universally good solutions.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Type

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.kernels import Stationary
from gpflow.models import GPR, SGPR, SVGP, VGP, GPModel

from ...data import Dataset, split_dataset_by_fidelity
from ...space import Box, EncoderFunction, SearchSpace, one_hot_encoded_space, one_hot_encoder
from ...types import TensorType
from ..gpflow.models import GaussianProcessRegression
from ..interfaces import encode_dataset

# NOTE: As a static non-Tensor, this should really be a tf.constant (like the other constants).
# However, changing it breaks serialisation during the expected_improvement.pct.py notebook.
# See https://github.com/secondmind-labs/trieste/issues/833 for details.
KERNEL_LENGTHSCALE = tf.cast(0.2, dtype=gpflow.default_float())
"""
Default value of the kernel lengthscale parameter.
"""


KERNEL_PRIOR_SCALE = tf.constant(1.0, dtype=gpflow.default_float())
"""
Default value of the scaling factor for the kernel lengthscale and variance parameters.
"""


CLASSIFICATION_KERNEL_VARIANCE_NOISE_FREE = tf.constant(100.0, dtype=gpflow.default_float())
"""
Default value of the kernel variance parameter for classification models in the noise free case.
"""


CLASSIFICATION_KERNEL_VARIANCE = tf.constant(1.0, dtype=gpflow.default_float())
"""
Default value of the kernel variance parameter for classification models.
"""


MAX_NUM_INDUCING_POINTS = tf.constant(500, dtype=tf.int32)
"""
Default maximum number of inducing points.
"""


NUM_INDUCING_POINTS_PER_DIM = tf.constant(25, dtype=tf.int32)
"""
Default number of inducing points per dimension of the search space.
"""


SIGNAL_NOISE_RATIO_LIKELIHOOD = tf.constant(10, dtype=gpflow.default_float())
"""
Default value used for initializing (noise) variance parameter of the likelihood function.
If user does not specify it, the noise variance is set to maintain the signal to noise ratio
determined by this default value. Signal variance in the kernel is set to the empirical variance.
"""


def build_gpr(
    data: Dataset,
    search_space: Optional[SearchSpace] = None,
    kernel_priors: bool = True,
    likelihood_variance: Optional[float] = None,
    trainable_likelihood: bool = False,
    kernel: Optional[gpflow.kernels.Kernel] = None,
    encoder: EncoderFunction | None = None,
    space_encoder: Callable[[SearchSpace], SearchSpace] | None = None,
) -> GPR:
    """
    Build a :class:`~gpflow.models.GPR` model with sensible initial parameters and
    priors. By default, we use :class:`~gpflow.kernels.Matern52` kernel and
    :class:`~gpflow.mean_functions.Constant` mean function in the model. We found the default
    configuration used here to work well in most situations, but it should not be taken as a
    universally good solution.

    We set priors for kernel hyperparameters by default in order to stabilize model fitting. We
    found the priors below to be highly effective for objective functions defined over the unit
    hypercube. They do seem to work for other search space sizes, but we advise caution when using
    them in such search spaces. Using priors allows for using maximum a posteriori estimate of
    these kernel parameters during model fitting.

    Note that although we scale parameters as a function of the size of the search space, ideally
    inputs should be normalised to the unit hypercube before building a model.

    :param data: Dataset from the initial design, used for estimating the variance of observations.
    :param search_space: Search space for performing Bayesian optimization, used for scaling the
        parameters. Required unless a kernel is passed.
    :param kernel_priors: If set to `True` (default) priors are set for kernel parameters (variance
        and lengthscale).
    :param likelihood_variance: Likelihood (noise) variance parameter can be optionally set to a
        certain value. If left unspecified (default), the noise variance is set to maintain the
        signal to noise ratio of value given by ``SIGNAL_NOISE_RATIO_LIKELIHOOD``, where signal
        variance in the kernel is set to the empirical variance.
    :param trainable_likelihood: If set to `True` Gaussian likelihood parameter is set to
        non-trainable. By default set to `False`.
    :param kernel: The kernel to use in the model, defaults to letting the function set up a
        :class:`~gpflow.kernels.Matern52` kernel.
    :param encoder: Encoder with which to transform the dataset before training. Defaults to
        one_hot_encoder if the search_space is specified.
    :param space_encoder: Encoder with which to transform search_space before generating a kernel.
        Defaults to one_hot_encoded_space.
    :return: A :class:`~gpflow.models.GPR` model.
    """
    if search_space is not None:
        encoder = one_hot_encoder(search_space) if encoder is None else encoder
        space_encoder = one_hot_encoded_space if space_encoder is None else space_encoder
        search_space = space_encoder(search_space)

    if encoder is not None:
        data = encode_dataset(data, encoder)

    empirical_mean, empirical_variance, _ = _get_data_stats(data)

    if kernel is None:
        if search_space is None:
            raise ValueError(
                "'build_gpr' function requires one of 'search_space' or 'kernel' arguments,"
                " but got neither"
            )
        kernel = _get_kernel(empirical_variance, search_space, kernel_priors, kernel_priors)
    mean = _get_mean_function(empirical_mean)

    assert isinstance(kernel, gpflow.kernels.Kernel)
    model = gpflow.models.GPR(data.astuple(), kernel, mean)

    _set_gaussian_likelihood_variance(model, empirical_variance, likelihood_variance)
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
    :class:`~gpflow.mean_functions.Constant` mean function in the model. We found the default
    configuration used here to work well in most situation, but it should not be taken as a
    universally good solution.

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
        signal to noise ratio of value given by ``SIGNAL_NOISE_RATIO_LIKELIHOOD``, where signal
        variance in the kernel is set to the empirical variance.
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
    mean = _get_mean_function(empirical_mean)

    inducing_points = gpflow.inducing_variables.InducingPoints(
        _get_inducing_points(search_space, num_inducing_points)
    )

    model = SGPR(data.astuple(), kernel, inducing_points, mean_function=mean)

    _set_gaussian_likelihood_variance(model, empirical_variance, likelihood_variance)
    gpflow.set_trainable(model.likelihood, trainable_likelihood)

    gpflow.set_trainable(model.inducing_variable, trainable_inducing_points)

    return model


def build_vgp_classifier(
    data: Dataset,
    search_space: SearchSpace,
    kernel_priors: bool = True,
    noise_free: bool = False,
    kernel_variance: Optional[float] = None,
) -> VGP:
    """
    Build a :class:`~gpflow.models.VGP` binary classification model with sensible initial
    parameters and priors. We use :class:`~gpflow.kernels.Matern52` kernel and
    :class:`~gpflow.mean_functions.Constant` mean function in the model. We found the default
    configuration used here to work well in most situation, but it should not be taken as a
    universally good solution.

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
        tf.debugging.assert_positive(kernel_variance)
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

    model_likelihood = gpflow.likelihoods.Bernoulli()
    kernel = _get_kernel(variance, search_space, kernel_priors, add_prior_to_variance)
    mean = _get_mean_function(tf.constant(0.0, dtype=gpflow.default_float()))

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
    priors. Both regression and binary classification models are
    available. We use :class:`~gpflow.kernels.Matern52` kernel and
    :class:`~gpflow.mean_functions.Constant` mean function in the model. We found the default
    configuration used here to work well in most situation, but it should not be taken as a
    universally good solution.

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
        signal to noise ratio of value given by ``SIGNAL_NOISE_RATIO_LIKELIHOOD``, where signal
        variance in the kernel is set to the empirical variance. This argument is ignored in the
        classification case.
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
        empirical_mean = tf.constant(0.0, dtype=gpflow.default_float())
        model_likelihood = gpflow.likelihoods.Bernoulli()
    else:
        model_likelihood = gpflow.likelihoods.Gaussian()

    kernel = _get_kernel(empirical_variance, search_space, kernel_priors, kernel_priors)
    mean = _get_mean_function(empirical_mean)

    inducing_points = _get_inducing_points(search_space, num_inducing_points)

    model = SVGP(
        kernel,
        model_likelihood,
        inducing_points,
        mean_function=mean,
        num_data=num_data_points,
    )

    if not classification:
        _set_gaussian_likelihood_variance(model, empirical_variance, likelihood_variance)
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
    lengthscales = _get_lengthscales(search_space)

    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=lengthscales)

    if add_prior_to_lengthscale:
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(lengthscales), KERNEL_PRIOR_SCALE
        )
    if add_prior_to_variance:
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.math.log(variance), KERNEL_PRIOR_SCALE
        )

    return kernel


def _get_lengthscales(search_space: SearchSpace) -> TensorType:
    lengthscales = (
        KERNEL_LENGTHSCALE
        * (search_space.upper - search_space.lower)
        * math.sqrt(search_space.dimension)
    )
    search_space_collapsed = tf.equal(search_space.upper, search_space.lower)
    lengthscales = tf.where(
        search_space_collapsed, tf.constant(1.0, dtype=gpflow.default_float()), lengthscales
    )
    return lengthscales


def _get_mean_function(mean: TensorType) -> gpflow.mean_functions.MeanFunction:
    mean_function = gpflow.mean_functions.Constant(mean)

    return mean_function


def _set_gaussian_likelihood_variance(
    model: GPModel, variance: TensorType, likelihood_variance: Optional[float]
) -> None:
    if likelihood_variance is None:
        noise_variance = variance / SIGNAL_NOISE_RATIO_LIKELIHOOD**2
    else:
        tf.debugging.assert_positive(likelihood_variance)
        noise_variance = tf.cast(likelihood_variance, dtype=gpflow.default_float())

    model.likelihood.variance = gpflow.base.Parameter(
        noise_variance, transform=gpflow.utilities.positive(lower=1e-12)
    )


def _get_inducing_points(
    search_space: SearchSpace, num_inducing_points: Optional[int]
) -> TensorType:
    if num_inducing_points is not None:
        tf.debugging.assert_positive(num_inducing_points)
    else:
        num_inducing_points = min(
            MAX_NUM_INDUCING_POINTS, NUM_INDUCING_POINTS_PER_DIM * search_space.dimension
        )
    if isinstance(search_space, Box):
        inducing_points = search_space.sample_sobol(num_inducing_points)
    else:
        inducing_points = search_space.sample(num_inducing_points)

    return inducing_points


def build_multifidelity_autoregressive_models(
    dataset: Dataset,
    num_fidelities: int,
    input_search_space: SearchSpace,
    likelihood_variance: float = 1e-6,
    kernel_priors: bool = False,
    trainable_likelihood: bool = False,
) -> Sequence[GaussianProcessRegression]:
    """
    Build the individual GPR models required for constructing an MultifidelityAutoregressive model
    with `num_fidelities` fidelities.

    :param dataset: Dataset of points with which to initialise the individual models,
        where the final column of the final dimension of the query points contains the fidelity
    :param num_fidelities: Number of fidelities desired for the MultifidelityAutoregressive model
    :param input_search_space: The input search space of the models
    :return: List of initialised GPR models
    """

    # Split data into fidelities
    data = split_dataset_by_fidelity(dataset=dataset, num_fidelities=num_fidelities)

    _validate_multifidelity_data_modellable(data, num_fidelities)

    gprs = [
        GaussianProcessRegression(
            build_gpr(
                data[fidelity],
                input_search_space,
                likelihood_variance=likelihood_variance,
                kernel_priors=kernel_priors,
                trainable_likelihood=trainable_likelihood,
            )
        )
        for fidelity in range(num_fidelities)
    ]

    return gprs


def build_multifidelity_nonlinear_autoregressive_models(
    dataset: Dataset,
    num_fidelities: int,
    input_search_space: SearchSpace,
    kernel_base_class: Type[Stationary] = gpflow.kernels.Matern32,
    kernel_priors: bool = True,
    trainable_likelihood: bool = False,
) -> Sequence[GaussianProcessRegression]:
    """
    Build models for training the trieste.models.gpflow.MultifidelityNonlinearAutoregressive` model

    Builds a basic Matern32 kernel for the lowest fidelity, and the custom kernel described in
    :cite:`perdikaris2017nonlinear` for the higher fidelities, which also have an extra input
    dimension. Note that the initial data that the models with fidelity greater than 0 are
    initialised with contain dummy data in this extra dimension, and so an `update` of the
    `MultifidelityNonlinearAutoregressive` is required to propagate real data through to these
    models.

    :param dataset: The dataset to use to initialise the models
    :param num_fidelities: The number of fidelities to model
    :param input_search_space: the search space, used to initialise the kernel parameters
    :param kernel_base_class: a stationary kernel type
    :param kernel_priors: If set to `True` (default) priors are set for kernel parameters (variance
        and lengthscale).
    :return: gprs: A list containing gprs that can be used for the multifidelity model
    """
    # Split data into fidelities
    data = split_dataset_by_fidelity(dataset=dataset, num_fidelities=num_fidelities)

    _validate_multifidelity_data_modellable(data, num_fidelities)

    # Input dim requires excluding fidelity row
    input_dim = dataset.query_points.shape[1] - 1

    # Create kernels
    kernels = _create_multifidelity_nonlinear_autoregressive_kernels(
        kernel_base_class,
        num_fidelities,
        input_dim,
        input_search_space,
        kernel_priors,
        kernel_priors,
    )

    # Initialise low fidelity GP
    gprs = [
        GaussianProcessRegression(
            build_gpr(
                data[0],
                search_space=input_search_space,  # This isn't actually used when we pass a kernel
                kernel=kernels[0],
                likelihood_variance=1e-6,
                trainable_likelihood=trainable_likelihood,
            )
        )
    ]

    for fidelity in range(1, num_fidelities):
        # Get query points for this fidelity
        qps = data[fidelity].query_points
        samples_column = tf.random.normal([qps.shape[0], 1], dtype=tf.float64)

        augmented_qps = tf.concat([qps, samples_column], axis=1)
        augmented_dataset = Dataset(augmented_qps, data[fidelity].observations)
        gprs.append(
            GaussianProcessRegression(
                build_gpr(
                    augmented_dataset,
                    input_search_space,  # This isn't actually used when we pass a kernel
                    kernel=kernels[fidelity],
                    likelihood_variance=1e-6,
                )
            )
        )

    return gprs


def _validate_multifidelity_data_modellable(data: Sequence[Dataset], num_fidelities: int) -> None:
    if num_fidelities < 2:
        raise ValueError(
            "Invalid number of fidelities to build Multifidelity model for,"
            f" need at least 2 fidelities, got {num_fidelities}"
        )

    for i, fidelity_data in enumerate(data):
        if len(fidelity_data) < 2:
            raise ValueError(
                f"Not enough data to create model for fidelity {i},"
                f" need at least 2 datapoints, got {len(fidelity_data)}"
            )


def _create_multifidelity_nonlinear_autoregressive_kernels(
    kernel_base_class: Type[Stationary],
    n_fidelities: int,
    n_input_dims: int,
    search_space: SearchSpace,
    add_prior_to_lengthscale: bool,
    add_prior_to_variance: bool,
) -> Sequence[Stationary]:
    dims = list(range(n_input_dims + 1))
    lengthscales = _get_lengthscales(search_space)

    scale_lengthscale = 1.0
    kernels = [kernel_base_class(lengthscales=lengthscales)]

    for _ in range(1, n_fidelities):
        interaction_kernel = kernel_base_class(lengthscales=lengthscales, active_dims=dims[:-1])
        scale_kernel = kernel_base_class(lengthscales=scale_lengthscale, active_dims=[dims[-1]])
        bias_kernel = kernel_base_class(lengthscales=lengthscales, active_dims=dims[:-1])
        gpflow.set_trainable(scale_kernel.variance, False)

        if add_prior_to_lengthscale:
            interaction_kernel.lengthscales.prior = tfp.distributions.LogNormal(
                tf.math.log(lengthscales), KERNEL_PRIOR_SCALE
            )
            bias_kernel.lengthscales.prior = tfp.distributions.LogNormal(
                tf.math.log(lengthscales), KERNEL_PRIOR_SCALE
            )
            scale_kernel.lengthscales.prior = tfp.distributions.LogNormal(
                tf.math.log(tf.cast(scale_lengthscale, dtype=gpflow.default_float())),
                KERNEL_PRIOR_SCALE,
            )

        if add_prior_to_variance:
            interaction_kernel.variance.prior = tfp.distributions.LogNormal(
                tf.constant(0.0, dtype=gpflow.default_float()), KERNEL_PRIOR_SCALE
            )
            bias_kernel.variance.prior = tfp.distributions.LogNormal(
                tf.constant(0.0, dtype=gpflow.default_float()), KERNEL_PRIOR_SCALE
            )

        kernels.append(interaction_kernel * scale_kernel + bias_kernel)

    return kernels
