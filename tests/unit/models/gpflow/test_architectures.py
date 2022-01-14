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
In this module, we test that we are wrapping GPflux architectures correctly, leading to the same
model.
"""

from __future__ import annotations

from typing import Optional

import gpflow
import numpy.testing as npt
import pytest
from gpflow.models import GPR, SGPR, SVGP, VGP, GPModel

from tests.util.misc import mk_dataset
from tests.util.models.gpflow.models import mock_data
from trieste.models.gpflow.architectures import (
    CLASSIFICATION_KERNEL_VARIANCE,
    CLASSIFICATION_KERNEL_VARIANCE_NOISE_FREE,
    MAX_NUM_INDUCING_POINTS,
    NUM_INDUCING_POINTS_PER_DIM,
    SNR_LIKELIHOOD,
    _get_data_stats,
    build_gpr,
    build_sgpr,
    build_svgp,
    build_vgp,
)
from trieste.space import Box, SearchSpace

# test discrete search space!


@pytest.mark.parametrize("kernel_priors", [True, False])
@pytest.mark.parametrize("likelihood_variance", [None, 0.1, 10.0])
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


@pytest.mark.parametrize("kernel_priors", [True, False])
@pytest.mark.parametrize("likelihood_variance", [None, 0.1, 10.0])
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


@pytest.mark.parametrize("kernel_priors", [True, False])
@pytest.mark.parametrize("noise_free", [True, False])
@pytest.mark.parametrize("kernel_variance", [None, 0.1, 10.0])
def test_build_vgp_returns_correct_model(
    kernel_priors: bool, noise_free: bool, kernel_variance: Optional[float]
) -> None:
    qp, obs = mock_data()
    data = mk_dataset(qp, obs)
    search_space = Box([0.0], [1.0]) ** qp.shape[-1]

    model = build_vgp(data, search_space, kernel_priors, noise_free, kernel_variance)

    # breakpoint()
    # basics
    assert isinstance(model, VGP)
    assert model.data == (qp, obs)

    # check the likelihood
    _check_likelihood(model, True, None, None, None)

    # check the mean function
    _check_mean_function(model, True, None)

    # check the kernel
    _check_kernel(model, True, kernel_variance, None, kernel_priors, noise_free)


@pytest.mark.parametrize("classification", [True, False])
@pytest.mark.parametrize("kernel_priors", [True, False])
@pytest.mark.parametrize("likelihood_variance", [None, 0.1, 10.0])
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


def _check_likelihood(
    model: GPModel,
    classification: bool,
    likelihood_variance: Optional[float],
    empirical_variance: Optional[float],
    trainable_likelihood: bool,
) -> None:
    if classification:
        assert isinstance(model.likelihood, gpflow.likelihoods.Bernoulli)
    else:
        assert isinstance(model.likelihood, gpflow.likelihoods.Gaussian)
        if likelihood_variance is not None:
            npt.assert_allclose(model.likelihood.variance, likelihood_variance, rtol=1e-6)
        else:
            npt.assert_allclose(
                model.likelihood.variance, empirical_variance / SNR_LIKELIHOOD ** 2, rtol=1e-6
            )
        assert model.likelihood.variance.trainable == trainable_likelihood


def _check_mean_function(
    model: GPModel, classification: bool, empirical_mean: Optional[float]
) -> None:
    assert isinstance(model.mean_function, gpflow.mean_functions.Constant)
    if classification:
        npt.assert_allclose(model.mean_function.parameters[0], 0.0, rtol=1e-6)
    else:
        npt.assert_allclose(model.mean_function.parameters[0], empirical_mean, rtol=1e-6)


def _check_kernel(
    model: GPModel,
    classification: bool,
    kernel_variance: Optional[float],
    empirical_variance: Optional[float],
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
        variance = empirical_variance
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
    num_inducing_points: int,
    trainable_inducing_points: bool,
) -> None:
    if num_inducing_points is None:
        num_inducing_points = min(
            MAX_NUM_INDUCING_POINTS, NUM_INDUCING_POINTS_PER_DIM * search_space.dimension
        )
    assert model.inducing_variable.num_inducing == num_inducing_points
    assert model.inducing_variable.parameters[0].trainable == trainable_inducing_points
