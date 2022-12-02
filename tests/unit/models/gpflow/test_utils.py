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
import operator
from typing import cast

import gpflow
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import random_seed
from tests.util.models.gpflow.models import ModelFactoryType
from tests.util.models.models import fnc_2sin_x_over_3
from trieste.data import Dataset
from trieste.models import TrainableProbabilisticModel
from trieste.models.gpflow import (
    check_optimizer,
    randomize_hyperparameters,
    squeeze_hyperparameters,
)
from trieste.models.optimizer import BatchOptimizer, Optimizer


def test_gaussian_process_deep_copyable(gpflow_interface_factory: ModelFactoryType) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model, _ = gpflow_interface_factory(x, fnc_2sin_x_over_3(x))
    model_copy = copy.deepcopy(model)
    x_predict = tf.constant([[50.5]], gpflow.default_float())

    # check deepcopy predicts same values as original
    mean_f, variance_f = model.predict(x_predict)
    mean_f_copy, variance_f_copy = model_copy.predict(x_predict)
    npt.assert_equal(mean_f, mean_f_copy)
    npt.assert_equal(variance_f, variance_f_copy)

    # check that updating the original doesn't break or change the deepcopy
    x_new = tf.concat([x, tf.constant([[10.0], [11.0]], dtype=gpflow.default_float())], 0)
    new_data = Dataset(x_new, fnc_2sin_x_over_3(x_new))
    cast(TrainableProbabilisticModel, model).update(new_data)
    model.optimize(new_data)

    mean_f_updated, variance_f_updated = model.predict(x_predict)
    mean_f_copy_updated, variance_f_copy_updated = model_copy.predict(x_predict)
    npt.assert_equal(mean_f_copy_updated, mean_f_copy)
    npt.assert_equal(variance_f_copy_updated, variance_f_copy)
    npt.assert_array_compare(operator.__ne__, mean_f_updated, mean_f)
    npt.assert_array_compare(operator.__ne__, variance_f_updated, variance_f)


@random_seed
@pytest.mark.parametrize("compile", [False, True])
def test_randomize_hyperparameters_randomizes_kernel_parameters_with_priors(
    dim: int, compile: bool
) -> None:
    kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=[0.2] * dim)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        loc=tf.math.log(kernel.lengthscales), scale=1.0
    )
    compiler = tf.function if compile else lambda x: x
    compiler(randomize_hyperparameters)(kernel)

    npt.assert_allclose(1.0, kernel.variance)
    npt.assert_array_equal(dim, kernel.lengthscales.shape)
    npt.assert_raises(AssertionError, npt.assert_allclose, [0.2] * dim, kernel.lengthscales)
    assert len(np.unique(kernel.lengthscales)) == dim


@random_seed
@pytest.mark.parametrize("compile", [False, True])
def test_randomize_hyperparameters_randomizes_kernel_parameters_with_const_priors(
    dim: int, compile: bool
) -> None:
    kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=[0.2] * dim)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        loc=tf.math.log(0.2), scale=1.0  # constant loc should be applied to every dimension
    )
    compiler = tf.function if compile else lambda x: x
    compiler(randomize_hyperparameters)(kernel)

    npt.assert_allclose(1.0, kernel.variance)
    npt.assert_array_equal(dim, kernel.lengthscales.shape)
    npt.assert_raises(AssertionError, npt.assert_allclose, [0.2] * dim, kernel.lengthscales)
    assert len(np.unique(kernel.lengthscales)) == dim


@random_seed
def test_randomize_hyperparameters_randomizes_constrained_kernel_parameters(dim: int) -> None:
    kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=[0.2] * dim)
    upper = tf.cast([10.0] * dim, dtype=tf.float64)
    lower = upper / 100
    kernel.lengthscales = gpflow.Parameter(
        kernel.lengthscales, transform=tfp.bijectors.Sigmoid(low=lower, high=upper)
    )

    randomize_hyperparameters(kernel)

    npt.assert_allclose(1.0, kernel.variance)
    npt.assert_array_equal(dim, kernel.lengthscales.shape)
    npt.assert_raises(AssertionError, npt.assert_allclose, [0.2] * dim, kernel.lengthscales)


@random_seed
def test_randomize_hyperparameters_randomizes_kernel_parameters_with_constraints_or_priors(
    dim: int,
) -> None:
    kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=[0.2] * dim)
    upper = tf.cast([10.0] * dim, dtype=tf.float64)
    lower = upper / 100
    kernel.lengthscales = gpflow.Parameter(
        kernel.lengthscales, transform=tfp.bijectors.Sigmoid(low=lower, high=upper)
    )
    kernel.variance.prior = tfp.distributions.LogNormal(loc=np.float64(-2.0), scale=np.float64(1.0))

    randomize_hyperparameters(kernel)

    npt.assert_raises(AssertionError, npt.assert_allclose, 1.0, kernel.variance)
    npt.assert_array_equal(dim, kernel.lengthscales.shape)
    npt.assert_raises(AssertionError, npt.assert_allclose, [0.2] * dim, kernel.lengthscales)


@random_seed
def test_randomize_hyperparameters_samples_from_constraints_when_given_prior_and_constraint(
    dim: int,
) -> None:
    kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=[0.2] * dim)
    upper = tf.cast([0.5] * dim, dtype=tf.float64)

    lower = upper / 100
    kernel.lengthscales = gpflow.Parameter(
        kernel.lengthscales, transform=tfp.bijectors.Sigmoid(low=lower, high=upper)
    )
    kernel.lengthscales.prior = tfp.distributions.Uniform(low=10.0, high=100.0)

    kernel.variance.prior = tfp.distributions.LogNormal(loc=np.float64(-2.0), scale=np.float64(1.0))

    randomize_hyperparameters(kernel)

    npt.assert_array_less(kernel.lengthscales, [0.5] * dim)
    npt.assert_raises(AssertionError, npt.assert_allclose, [0.2] * dim, kernel.lengthscales)


@random_seed
def test_randomize_hyperparameters_samples_different_values_for_multi_dimensional_params() -> None:
    kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=[0.2, 0.2])
    upper = tf.cast([10.0] * 2, dtype=tf.float64)
    lower = upper / 100
    kernel.lengthscales = gpflow.Parameter(
        kernel.lengthscales, transform=tfp.bijectors.Sigmoid(low=lower, high=upper)
    )
    randomize_hyperparameters(kernel)
    npt.assert_raises(
        AssertionError, npt.assert_allclose, kernel.lengthscales[0], kernel.lengthscales[1]
    )


@random_seed
def test_squeeze_sigmoid_hyperparameters() -> None:
    kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=[0.1 + 1e-3, 0.5 - 1e-3])
    upper = tf.cast([0.5, 0.5], dtype=tf.float64)
    lower = upper / 5.0
    kernel.lengthscales = gpflow.Parameter(
        kernel.lengthscales, transform=tfp.bijectors.Sigmoid(low=lower, high=upper)
    )
    squeeze_hyperparameters(kernel, alpha=0.1)
    npt.assert_array_almost_equal(kernel.lengthscales, [0.1 + 4e-2, 0.5 - 4e-2])


@random_seed
def test_squeeze_softplus_hyperparameters() -> None:
    lik = gpflow.likelihoods.Gaussian(variance=1.01e-6)
    squeeze_hyperparameters(lik, epsilon=0.2)
    npt.assert_array_almost_equal(lik.variance, 0.2 + 1e-6)


@random_seed
def test_squeeze_raises_for_invalid_epsilon() -> None:
    lik = gpflow.likelihoods.Gaussian(variance=1.01e-6)
    with pytest.raises(ValueError):
        squeeze_hyperparameters(lik, epsilon=-1.0)


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.1])
def test_squeeze_raises_for_invalid_alpha(alpha: float) -> None:
    kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=[0.2, 0.2])
    upper = tf.cast([0.5, 0.5], dtype=tf.float64)
    lower = upper / 5.0
    kernel.lengthscales = gpflow.Parameter(
        kernel.lengthscales, transform=tfp.bijectors.Sigmoid(low=lower, high=upper)
    )
    with pytest.raises(ValueError):
        squeeze_hyperparameters(kernel, alpha)


def test_check_optimizer_raises_for_invalid_optimizer_wrapper_combination() -> None:

    with pytest.raises(ValueError):
        optimizer1 = BatchOptimizer(gpflow.optimizers.Scipy())
        check_optimizer(optimizer1)

    with pytest.raises(ValueError):
        optimizer2 = Optimizer(tf.optimizers.Adam())
        check_optimizer(optimizer2)
