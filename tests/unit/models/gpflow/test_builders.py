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
In this module, we test that we are wrapping GPflow architectures correctly, leading to the same
model.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
from gpflow.models import GPR, SGPR, SVGP, VGP, GPModel

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, mk_dataset, quadratic
from tests.util.models.gpflow.models import mock_data
from trieste.data import Dataset
from trieste.models.gpflow.builders import (
    CLASSIFICATION_KERNEL_VARIANCE,
    CLASSIFICATION_KERNEL_VARIANCE_NOISE_FREE,
    KERNEL_LENGTHSCALE,
    MAX_NUM_INDUCING_POINTS,
    NUM_INDUCING_POINTS_PER_DIM,
    SIGNAL_NOISE_RATIO_LIKELIHOOD,
    _get_data_stats,
    build_gpr,
    build_multifidelity_autoregressive_models,
    build_multifidelity_nonlinear_autoregressive_models,
    build_sgpr,
    build_svgp,
    build_vgp_classifier,
)
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.space import Box, DiscreteSearchSpace, SearchSpace
from trieste.types import TensorType


@pytest.mark.parametrize("kernel_priors", [True, False])
@pytest.mark.parametrize("likelihood_variance", [None, 1e-10, 10.0])
@pytest.mark.parametrize("trainable_likelihood", [True, False])
def test_build_gpr_returns_correct_model(
    kernel_priors: bool, likelihood_variance: Optional[float], trainable_likelihood: bool
) -> None:
    qp, obs = mock_data()
    data = mk_dataset(qp, obs)
    search_space = Box([0.0], [1.0]) ** qp.shape[-1]

    model = build_gpr(data, search_space, kernel_priors, likelihood_variance, trainable_likelihood)

    empirical_mean, empirical_variance, _ = _get_data_stats(data)

    # basics
    assert isinstance(model, GPR)
    assert model.data == (qp, obs)

    # check the likelihood
    _check_likelihood(model, False, likelihood_variance, empirical_variance, trainable_likelihood)

    # check the mean function
    _check_mean_function(model, False, empirical_mean)

    # check the kernel
    _check_kernel(model, False, None, empirical_variance, kernel_priors, False)


@pytest.mark.parametrize("likelihood_variance", [-1, 0.0])
def test_build_gpr_raises_for_invalid_likelihood_variance(likelihood_variance: float) -> None:
    qp, obs = mock_data()
    data = mk_dataset(qp, obs)
    search_space = Box([0.0], [1.0]) ** qp.shape[-1]

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        build_gpr(data, search_space, likelihood_variance=likelihood_variance)


@pytest.mark.parametrize("kernel_priors", [True, False])
@pytest.mark.parametrize("likelihood_variance", [None, 1e-10, 10.0])
@pytest.mark.parametrize("trainable_likelihood", [True, False])
@pytest.mark.parametrize("num_inducing_points", [None, 3, 100])
@pytest.mark.parametrize("trainable_inducing_points", [True, False])
def test_build_sgpr_returns_correct_model(
    kernel_priors: bool,
    likelihood_variance: Optional[float],
    trainable_likelihood: bool,
    num_inducing_points: Optional[int],
    trainable_inducing_points: bool,
) -> None:
    qp, obs = mock_data()
    data = mk_dataset(qp, obs)
    search_space = Box([0.0], [1.0]) ** qp.shape[-1]

    model = build_sgpr(
        data,
        search_space,
        kernel_priors,
        likelihood_variance,
        trainable_likelihood,
        num_inducing_points,
        trainable_inducing_points,
    )

    empirical_mean, empirical_variance, _ = _get_data_stats(data)

    # basics
    assert isinstance(model, SGPR)
    assert model.data == (qp, obs)

    # check the likelihood
    _check_likelihood(model, False, likelihood_variance, empirical_variance, trainable_likelihood)

    # check the mean function
    _check_mean_function(model, False, empirical_mean)

    # check the kernel
    _check_kernel(model, False, None, empirical_variance, kernel_priors, False)

    # check the inducing points
    _check_inducing_points(model, search_space, num_inducing_points, trainable_inducing_points)


@pytest.mark.parametrize("likelihood_variance", [-1, 0.0])
def test_build_sgpr_raises_for_invalid_likelihood_variance(likelihood_variance: float) -> None:
    qp, obs = mock_data()
    data = mk_dataset(qp, obs)
    search_space = Box([0.0], [1.0]) ** qp.shape[-1]

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        build_sgpr(data, search_space, likelihood_variance=likelihood_variance)


@pytest.mark.parametrize("num_inducing_points", [-1, 0])
def test_build_sgpr_raises_for_invalid_num_inducing_points(num_inducing_points: int) -> None:
    qp, obs = mock_data()
    data = mk_dataset(qp, obs)
    search_space = Box([0.0], [1.0]) ** qp.shape[-1]

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        build_sgpr(data, search_space, num_inducing_points=num_inducing_points)


@pytest.mark.parametrize("kernel_priors", [True, False])
@pytest.mark.parametrize("noise_free", [True, False])
@pytest.mark.parametrize("kernel_variance", [None, 0.1, 10.0])
def test_build_vgp_classifier_returns_correct_model(
    kernel_priors: bool, noise_free: bool, kernel_variance: Optional[float]
) -> None:
    qp, obs = mock_data()
    data = mk_dataset(qp, obs)
    search_space = Box([0.0], [1.0]) ** qp.shape[-1]

    model = build_vgp_classifier(data, search_space, kernel_priors, noise_free, kernel_variance)

    # breakpoint()
    # basics
    assert isinstance(model, VGP)
    assert model.data == (qp, obs)

    # check the likelihood
    _check_likelihood(model, True, None, None, False)

    # check the mean function
    _check_mean_function(model, True, None)

    # check the kernel
    _check_kernel(model, True, kernel_variance, 0.0, kernel_priors, noise_free)


@pytest.mark.parametrize("kernel_variance", [-1, 0.0])
def test_build_vgp_classifier_raises_for_invalid_kernel_variance(kernel_variance: float) -> None:
    qp, obs = mock_data()
    data = mk_dataset(qp, obs)
    search_space = Box([0.0], [1.0]) ** qp.shape[-1]

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        build_vgp_classifier(data, search_space, kernel_variance=kernel_variance)


@pytest.mark.parametrize("classification", [True, False])
@pytest.mark.parametrize("kernel_priors", [True, False])
@pytest.mark.parametrize("likelihood_variance", [None, 1e-10, 10.0])
@pytest.mark.parametrize("trainable_likelihood", [True, False])
@pytest.mark.parametrize("num_inducing_points", [None, 3, 100])
@pytest.mark.parametrize("trainable_inducing_points", [True, False])
def test_build_svgp_returns_correct_model(
    classification: bool,
    kernel_priors: bool,
    likelihood_variance: Optional[float],
    trainable_likelihood: bool,
    num_inducing_points: Optional[int],
    trainable_inducing_points: bool,
) -> None:
    qp, obs = mock_data()
    data = mk_dataset(qp, obs)
    search_space = Box([0.0], [1.0]) ** qp.shape[-1]

    model = build_svgp(
        data,
        search_space,
        classification,
        kernel_priors,
        likelihood_variance,
        trainable_likelihood,
        num_inducing_points,
        trainable_inducing_points,
    )

    empirical_mean, empirical_variance, _ = _get_data_stats(data)

    # basics
    assert isinstance(model, SVGP)

    # check the likelihood
    _check_likelihood(
        model, classification, likelihood_variance, empirical_variance, trainable_likelihood
    )

    # check the mean function
    _check_mean_function(model, classification, empirical_mean)

    # check the kernel
    _check_kernel(model, classification, None, empirical_variance, kernel_priors, False)

    # check the inducing points
    _check_inducing_points(model, search_space, num_inducing_points, trainable_inducing_points)


@pytest.mark.parametrize("likelihood_variance", [-1, 0.0])
def test_build_svgp_raises_for_invalid_likelihood_variance(likelihood_variance: float) -> None:
    qp, obs = mock_data()
    data = mk_dataset(qp, obs)
    search_space = Box([0.0], [1.0]) ** qp.shape[-1]

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        build_svgp(data, search_space, likelihood_variance=likelihood_variance)


@pytest.mark.parametrize("num_inducing_points", [-1, 0])
def test_build_svgp_raises_for_invalid_num_inducing_points(num_inducing_points: int) -> None:
    qp, obs = mock_data()
    data = mk_dataset(qp, obs)
    search_space = Box([0.0], [1.0]) ** qp.shape[-1]

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        build_svgp(data, search_space, num_inducing_points=num_inducing_points)


@pytest.mark.parametrize(
    "lower, upper",
    [([0.0, 0.0], [1.0, 10.0]), ([0.0, -1.0], [4.0, 2.0]), ([-10.0, -2.0], [-1.0, -1.0])],
)
@pytest.mark.parametrize("builder", [build_gpr, build_sgpr, build_svgp, build_vgp_classifier])
def test_builder_returns_correct_lengthscales_for_unequal_box_bounds(
    lower: List[float], upper: List[float], builder: Any
) -> None:
    search_space = Box(lower, upper)
    qp = search_space.sample(10)
    data = mk_dataset(qp, quadratic(qp))

    model = builder(data, search_space)

    expected_lengthscales = (
        KERNEL_LENGTHSCALE
        * (search_space.upper - search_space.lower)
        * math.sqrt(search_space.dimension)
    )

    npt.assert_allclose(model.kernel.lengthscales, expected_lengthscales, rtol=1e-6)


@pytest.mark.parametrize(
    "points",
    [
        ([[0.0, 0.0], [1.0, 10.0]]),
        ([[0.0, -1.0], [4.0, 2.0]]),
        ([[-10.0, -2.0], [-1.0, -1.0]]),
        ([[0.0, 1.0], [2.0, 1.0]]),
        ([[0.0, 1.0], [0.0, 10.0]]),
    ],
)
@pytest.mark.parametrize("builder", [build_gpr, build_sgpr, build_svgp, build_vgp_classifier])
def test_builder_returns_correct_lengthscales_for_unequal_discrete_bounds(
    points: List[List[float]], builder: Any
) -> None:
    search_space = DiscreteSearchSpace(tf.constant(points, dtype=tf.float64))
    qp = search_space.sample(10)
    data = mk_dataset(qp, quadratic(qp))
    # breakpoint()
    model = builder(data, search_space)

    expected_lengthscales = (
        KERNEL_LENGTHSCALE
        * (search_space.upper - search_space.lower)
        * math.sqrt(search_space.dimension)
    )
    search_space_collapsed = tf.equal(search_space.upper, search_space.lower)
    expected_lengthscales = tf.where(
        search_space_collapsed, tf.cast(1.0, dtype=gpflow.default_float()), expected_lengthscales
    )

    npt.assert_allclose(model.kernel.lengthscales, expected_lengthscales, rtol=1e-6)


@pytest.mark.parametrize("model_type", ("linear", "nonlinear"))
def test_build_multifidelity_builds_correct_n_gprs(model_type: str) -> None:
    dataset = Dataset(
        tf.Variable(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 1.0], [4.0, 2.0], [5.0, 0.0]],
            dtype=tf.float64,
        ),
        tf.Variable([[2.0], [4.0], [6.0], [8.0], [10.0], [12.0]], dtype=tf.float64),
    )
    search_space = Box([0.0], [10.0])

    if model_type == "linear":
        gprs = build_multifidelity_autoregressive_models(dataset, 3, search_space)
    else:
        gprs = build_multifidelity_nonlinear_autoregressive_models(dataset, 3, search_space)

    assert len(gprs) == 3
    for gpr in gprs:
        assert isinstance(gpr, GaussianProcessRegression)


@pytest.mark.parametrize("model_type", ("linear", "nonlinear"))
def test_build_multifidelity_raises_for_bad_fidelity(model_type: str) -> None:
    dataset = Dataset(
        tf.Variable(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 1.0], [4.0, 2.0], [5.0, 0.0]],
            dtype=tf.float64,
        ),
        tf.Variable([[2.0], [4.0], [6.0], [8.0], [10.0], [12.0]], dtype=tf.float64),
    )
    search_space = Box([0.0], [10.0])
    with pytest.raises(ValueError):
        if model_type == "linear":
            build_multifidelity_autoregressive_models(dataset, -1, search_space)
        else:
            build_multifidelity_nonlinear_autoregressive_models(dataset, -1, search_space)


@pytest.mark.parametrize(
    "query_points,observations",
    (
        (
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 1.0], [4.0, 2.0]],  # Only 1 point for fid 0
            [[2.0], [4.0], [6.0], [8.0], [10.0]],
        ),
        (
            [[0.0, 0.0], [2.0, 2.0], [4.0, 2.0], [5.0, 0.0]],  # Missing middle fid entirely
            [[2.0], [4.0], [6.0], [8.0]],
        ),
        (
            [[0.0, 0.0], [2.0, 1.0], [4.0, 1.0], [5.0, 0.0]],  # 2 fid data, but fid set as 3
            [[2.0], [4.0], [6.0], [8.0]],
        ),
    ),
)
@pytest.mark.parametrize("model_type", ("linear", "nonlinear"))
def test_build_multifidelity_raises_not_enough_datapoints(
    query_points: TensorType, observations: TensorType, model_type: str
) -> None:
    dataset = Dataset(
        tf.Variable(
            query_points,
            dtype=tf.float64,
        ),
        tf.Variable(observations, dtype=tf.float64),
    )
    search_space = Box([0.0], [10.0])

    with pytest.raises(ValueError):
        if model_type == "linear":
            build_multifidelity_autoregressive_models(dataset, 3, search_space)
        else:
            build_multifidelity_nonlinear_autoregressive_models(dataset, 3, search_space)


@pytest.mark.parametrize("model_type", ("linear", "nonlinear"))
def test_build_multifidelity_raises_not_multifidelity_data(
    model_type: str,
) -> None:
    dataset = Dataset(
        tf.Variable(
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            dtype=tf.float64,
        ),
        tf.Variable([[2.0], [4.0], [6.0], [8.0], [10.0]], dtype=tf.float64),
    )
    search_space = Box([0.0], [10.0])

    with pytest.raises(ValueError):
        if model_type == "linear":
            build_multifidelity_autoregressive_models(dataset, 3, search_space)
        else:
            build_multifidelity_nonlinear_autoregressive_models(dataset, 3, search_space)


def _check_likelihood(
    model: GPModel,
    classification: bool,
    likelihood_variance: Optional[float],
    empirical_variance: Optional[TensorType],
    trainable_likelihood: bool,
) -> None:
    if classification:
        assert isinstance(model.likelihood, gpflow.likelihoods.Bernoulli)
    else:
        assert isinstance(model.likelihood, gpflow.likelihoods.Gaussian)
        if likelihood_variance is not None:
            npt.assert_allclose(
                tf.constant(model.likelihood.variance), likelihood_variance, rtol=1e-6
            )
        else:
            npt.assert_allclose(
                tf.constant(model.likelihood.variance),
                empirical_variance / SIGNAL_NOISE_RATIO_LIKELIHOOD**2,
                rtol=1e-6,
            )
        assert isinstance(model.likelihood.variance, gpflow.Parameter)
        assert model.likelihood.variance.trainable == trainable_likelihood


def _check_mean_function(
    model: GPModel, classification: bool, empirical_mean: Optional[TensorType]
) -> None:
    assert isinstance(model.mean_function, gpflow.mean_functions.Constant)
    if classification:
        npt.assert_allclose(model.mean_function.parameters[0], 0.0, rtol=1e-6)
    else:
        assert empirical_mean is not None
        npt.assert_allclose(model.mean_function.parameters[0], empirical_mean, rtol=1e-6)


def _check_kernel(
    model: GPModel,
    classification: bool,
    kernel_variance: Optional[float],
    empirical_variance: TensorType,
    kernel_priors: bool,
    noise_free: bool,
) -> None:
    assert isinstance(model.kernel, gpflow.kernels.Matern52)
    if classification:
        if kernel_variance is not None:
            variance = kernel_variance
        else:
            if noise_free:
                variance = CLASSIFICATION_KERNEL_VARIANCE_NOISE_FREE
            else:
                variance = CLASSIFICATION_KERNEL_VARIANCE
    else:
        variance = float(empirical_variance)
    npt.assert_allclose(model.kernel.variance, variance, rtol=1e-6)
    if kernel_priors:
        if noise_free:
            assert model.kernel.variance.prior is None
        else:
            assert model.kernel.variance.prior is not None
        assert model.kernel.lengthscales.prior is not None
    else:
        assert model.kernel.variance.prior is None
        assert model.kernel.lengthscales.prior is None


def _check_inducing_points(
    model: GPModel,
    search_space: SearchSpace,
    num_inducing_points: Optional[int],
    trainable_inducing_points: bool,
) -> None:
    if num_inducing_points is None:
        num_inducing_points = min(
            MAX_NUM_INDUCING_POINTS, NUM_INDUCING_POINTS_PER_DIM * search_space.dimension
        )
    assert model.inducing_variable.num_inducing == num_inducing_points
    assert model.inducing_variable.parameters[0].trainable == trainable_inducing_points
