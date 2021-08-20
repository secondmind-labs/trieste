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
In this module, we test the *behaviour* of trieste models against reference GPflux models (thus
implicitly assuming the latter are correct).

*NOTE:* Where GPflux models are used as the underlying model in an trieste model, we should
*not* test that the underlying model is used in any particular way. To do so would break
encapsulation. For example, we should *not* test that methods on the GPflux models are called
(except in the rare case that such behaviour is an explicitly documented behaviour of the
trieste model).
"""

from __future__ import annotations

import gpflow
import gpflux.encoders
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from gpflux.models import DeepGP
from gpflux.models.deep_gp import sample_dgp

from tests.util.misc import random_seed
from tests.util.models.models import fnc_3x_plus_10, fnc_2sin_x_over_3
from tests.util.models.gpflux.models import (
    single_layer_dgp_model,
    two_layer_dgp_model,
    two_layer_dgp_model_no_whitening,
)
from trieste.data import Dataset
from trieste.models.gpflux import DeepGaussianProcess
from trieste.models.optimizer import Optimizer, TFOptimizer, create_optimizer
from trieste.types import TensorType
from trieste.utils import jit


def test_dgp_raises_for_non_tf_optimizer() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    dgp = two_layer_dgp_model(x)
    optimizer = Optimizer(gpflow.optimizers.Scipy())

    with pytest.raises(ValueError):
        DeepGaussianProcess(dgp, optimizer=optimizer)


def test_dgp_raises_for_latent_variable_layer() -> None:
    num_data = 5

    encoder = gpflux.encoders.DirectlyParameterizedNormalDiag(num_data, 1)
    prior = tfp.distributions.MultivariateNormalDiag(np.zeros(1), np.ones(1))
    lv = gpflux.layers.LatentVariableLayer(prior, encoder)
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0] * 2)
    num_inducing = 5
    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.concatenate(
            [
                np.linspace(-1, 1, num_inducing).reshape(-1, 1),
                np.random.randn(num_inducing, 1),
            ],
            axis=1,
        )
    )
    gp_layer = gpflux.layers.GPLayer(
        kernel,
        inducing_variable,
        num_data=num_data,
        num_latent_gps=1,
        mean_function=gpflow.mean_functions.Zero(),
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.01))

    dgp = DeepGP([lv, gp_layer], likelihood_layer)

    with pytest.raises(ValueError):
        DeepGaussianProcess(dgp)


def test_dgp_raises_for_keras_layer() -> None:
    keras_layer_1 = tf.keras.layers.Dense(50, activation="relu")
    keras_layer_2 = tf.keras.layers.Dense(2, activation="relu")

    kernel = gpflow.kernels.SquaredExponential()
    num_inducing = 5
    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.concatenate(
            [
                np.random.randn(num_inducing, 2),
            ],
            axis=1,
        )
    )
    gp_layer = gpflux.layers.GPLayer(
        kernel,
        inducing_variable,
        num_data=5,
        num_latent_gps=1,
        mean_function=gpflow.mean_functions.Zero(),
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.01))

    dgp = DeepGP([keras_layer_1, keras_layer_2, gp_layer], likelihood_layer)

    with pytest.raises(ValueError):
        DeepGaussianProcess(dgp)


def test_dgp_warns_for_whitened_layers() -> None:
    num_data = 5
    num_inducing = num_data

    Z = np.linspace(-1, 1, num_inducing).reshape(-1, 1)
    kernel_1 = gpflow.kernels.SquaredExponential()
    inducing_variable_1 = gpflow.inducing_variables.InducingPoints(Z.copy())
    gp_layer_1 = gpflux.layers.GPLayer(
        kernel_1,
        inducing_variable_1,
        num_data=num_data,
        num_latent_gps=1,
        whiten=False,
    )

    kernel_2 = gpflow.kernels.SquaredExponential()
    inducing_variable_2 = gpflow.inducing_variables.InducingPoints(Z.copy())
    gp_layer_2 = gpflux.layers.GPLayer(
        kernel_2,
        inducing_variable_2,
        num_data=num_data,
        num_latent_gps=1,
        whiten=True,
    )

    dgp = DeepGP([gp_layer_1, gp_layer_2], gpflow.likelihoods.Gaussian(0.01))

    with pytest.warns(UserWarning,
                      match='Sampling cannot be currently used with whitening in layers'):
        DeepGaussianProcess(dgp)


def test_dgp_model_attribute() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    dgp = two_layer_dgp_model(x)
    model = DeepGaussianProcess(dgp)

    assert model.model is dgp


def test_dgp_update() -> None:
    x = tf.zeros([1, 4])
    dgp = two_layer_dgp_model(x)
    model = DeepGaussianProcess(dgp)

    assert model.model.num_data == 1

    for layer in model.model.f_layers:
        assert layer.num_data == 1

    model.update(Dataset(tf.zeros([5, 4]), tf.zeros([5, 1])))

    assert model.model.num_data == 5

    for layer in model.model.f_layers:
        assert layer.num_data == 5


@pytest.mark.parametrize(
    "new_data",
    [Dataset(tf.zeros([3, 5]), tf.zeros([3, 1])), Dataset(tf.zeros([3, 4]), tf.zeros([3, 2]))],
)
def test_dgp_update_raises_for_invalid_shapes(new_data: Dataset) -> None:
    x = tf.zeros([1, 4])
    dgp = two_layer_dgp_model(x)
    model = DeepGaussianProcess(dgp)

    with pytest.raises(ValueError):
        model.update(new_data)


def test_dgp_optimize_with_defaults() -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = fnc_2sin_x_over_3(x_observed)
    data = x_observed, y_observed
    dataset = Dataset(*data)
    optimizer = create_optimizer(tf.optimizers.Adam(), dict(max_iter=20))
    model = DeepGaussianProcess(two_layer_dgp_model(x_observed), optimizer=optimizer)
    elbo = model.model.elbo(data)
    model.optimize(dataset)
    assert model.model.elbo(data) > elbo


@pytest.mark.parametrize("batch_size", [10, 100])
def test_dgp_optimize(batch_size: int) -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = fnc_2sin_x_over_3(x_observed)
    data = x_observed, y_observed
    dataset = Dataset(*data)

    optimizer = create_optimizer(
        tf.optimizers.Adam(),
        dict(max_iter=10, batch_size=batch_size),
    )
    model = DeepGaussianProcess(two_layer_dgp_model(x_observed), optimizer=optimizer)
    elbo = model.model.elbo(data)
    model.optimize(dataset)
    assert model.model.elbo(data) > elbo


def test_dgp_loss() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    y = fnc_3x_plus_10(x)

    reference_model = two_layer_dgp_model(x)
    model = DeepGaussianProcess(two_layer_dgp_model(x))
    internal_model = model.model

    npt.assert_allclose(internal_model.elbo((x, y)), reference_model.elbo((x, y)), rtol=1e-6)


def test_dgp_predict() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())

    reference_model = single_layer_dgp_model(x)
    model = DeepGaussianProcess(single_layer_dgp_model(x))

    test_x = tf.constant([[2.5]], dtype=gpflow.default_float())

    ref_mean, ref_var = reference_model.predict_f(test_x)
    f_mean, f_var = model.predict(test_x)

    npt.assert_allclose(f_mean, ref_mean)
    npt.assert_allclose(f_var, ref_var)


@random_seed
def test_dgp_sample() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model = DeepGaussianProcess(
        two_layer_dgp_model_no_whitening(x),
        optimizer=TFOptimizer(tf.optimizers.Adam(), compile=True),
    )
    num_samples = 50
    test_x = tf.constant([[2.5]], dtype=gpflow.default_float())
    samples = model.sample(test_x, num_samples)

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2)

    assert samples.shape == [num_samples, 1, 1]

    reference_model = two_layer_dgp_model_no_whitening(x)

    sampler = sample_dgp(reference_model)

    @jit(apply=True)
    def get_samples(query_points: TensorType, num_samples: int) -> TensorType:
        samples = []
        for _ in range(num_samples):
            samples.append(sampler(query_points))
        return tf.stack(samples)

    ref_samples = get_samples(test_x, num_samples)

    ref_mean = tf.reduce_mean(ref_samples, axis=0)
    ref_variance = tf.reduce_mean((ref_samples - ref_mean) ** 2)

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, ref_mean, rtol=error)
    npt.assert_allclose(sample_variance, ref_variance, rtol=2 * error)
