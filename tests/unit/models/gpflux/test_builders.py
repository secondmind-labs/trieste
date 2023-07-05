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

import gpflow
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from gpflux.models import DeepGP

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, mk_dataset, quadratic
from trieste.models.gpflux.builders import (
    LIKELIHOOD_VARIANCE,
    MAX_NUM_INDUCING_POINTS,
    NUM_INDUCING_POINTS_PER_DIM,
    NUM_LAYERS,
    _get_data_stats,
    build_vanilla_deep_gp,
)
from trieste.space import Box


def test_build_vanilla_deep_gp_returns_correct_defaults() -> None:
    search_space = Box([0.0], [1.0]) ** 4
    x = search_space.sample(100)
    data = mk_dataset(x, quadratic(x))

    empirical_mean, empirical_variance, _ = _get_data_stats(data)

    num_inducing = min(
        MAX_NUM_INDUCING_POINTS, NUM_INDUCING_POINTS_PER_DIM * search_space.dimension
    )

    vanilla_deep_gp = build_vanilla_deep_gp(data, search_space)

    # basics
    assert isinstance(vanilla_deep_gp, DeepGP)
    assert len(vanilla_deep_gp.f_layers) == NUM_LAYERS

    # check mean function
    assert isinstance(vanilla_deep_gp.f_layers[-1].mean_function, gpflow.mean_functions.Constant)
    npt.assert_allclose(vanilla_deep_gp.f_layers[-1].mean_function.parameters[0], empirical_mean)

    # check kernel
    assert isinstance(vanilla_deep_gp.f_layers[-1].kernel.kernel, gpflow.kernels.RBF)
    npt.assert_allclose(vanilla_deep_gp.f_layers[-1].kernel.kernel.variance, empirical_variance)

    # check likelihood
    assert isinstance(vanilla_deep_gp.likelihood_layer.likelihood, gpflow.likelihoods.Gaussian)
    npt.assert_allclose(
        tf.constant(vanilla_deep_gp.likelihood_layer.likelihood.variance), LIKELIHOOD_VARIANCE
    )
    assert isinstance(vanilla_deep_gp.likelihood_layer.likelihood.variance, gpflow.Parameter)
    assert vanilla_deep_gp.likelihood_layer.likelihood.variance.trainable

    # inducing variables and scaling factor
    for layer in vanilla_deep_gp.f_layers:
        assert layer.inducing_variable.num_inducing == num_inducing


@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("likelihood_variance", [1e-5, 10.0])
@pytest.mark.parametrize("trainable_likelihood", [True, False])
@pytest.mark.parametrize("inner_layer_sqrt_factor", [1e-5, 10.0])
def test_build_vanilla_deep_gp_returns_correct_model(
    num_layers: int,
    likelihood_variance: float,
    trainable_likelihood: bool,
    inner_layer_sqrt_factor: bool,
) -> None:
    num_data = 10
    x = np.arange(num_data).reshape(-1, 1).astype(np.double)
    data = mk_dataset(x.tolist(), quadratic(x))
    search_space = Box([0.0], [10.0])

    num_inducing = num_data

    vanilla_deep_gp = build_vanilla_deep_gp(
        data,
        search_space,
        num_layers,
        num_inducing,
        inner_layer_sqrt_factor=inner_layer_sqrt_factor,
        likelihood_variance=likelihood_variance,
        trainable_likelihood=trainable_likelihood,
    )

    # check likelihood
    npt.assert_allclose(vanilla_deep_gp.likelihood_layer.likelihood.variance, likelihood_variance)
    assert vanilla_deep_gp.likelihood_layer.likelihood.variance.trainable == trainable_likelihood

    # comparison to the gpflux builder
    config = Config(
        num_inducing,
        inner_layer_sqrt_factor,
        likelihood_variance,
    )
    ref_deep_gp = build_constant_input_dim_deep_gp(x, num_layers=num_layers, config=config)
    npt.assert_equal(len(vanilla_deep_gp.f_layers), len(ref_deep_gp.f_layers))
    for i, layer in enumerate(vanilla_deep_gp.f_layers):
        ref_layer = ref_deep_gp.f_layers[i]
        npt.assert_allclose(
            tf.sort(layer.inducing_variable.inducing_variable.Z, axis=0),
            tf.sort(ref_layer.inducing_variable.inducing_variable.Z, axis=0),
        )
        npt.assert_allclose(layer.q_sqrt, ref_layer.q_sqrt)
    npt.assert_allclose(
        vanilla_deep_gp.likelihood_layer.likelihood.variance,
        ref_deep_gp.likelihood_layer.likelihood.variance,
    )


def test_build_vanilla_deep_gp_raises_for_incorrect_args() -> None:
    x = np.arange(10).reshape(-1, 1).astype(np.double)
    data = mk_dataset(x.tolist(), quadratic(x))
    search_space = Box([0.0], [10.0])

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        build_vanilla_deep_gp(data, search_space, 0)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        build_vanilla_deep_gp(data, search_space, num_inducing_points=0)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        build_vanilla_deep_gp(data, search_space, inner_layer_sqrt_factor=0)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        build_vanilla_deep_gp(data, search_space, likelihood_variance=0)


@pytest.mark.parametrize("multiplier", [1, 2, 5])
def test_build_vanilla_deep_gp_gives_correct_num_inducing_points_and_num_data(
    multiplier: int,
) -> None:
    num_data = 5
    x = np.arange(num_data).reshape(-1, 1).astype(np.double)
    data = mk_dataset(x.tolist(), quadratic(x))
    search_space = Box([0.0], [10.0])

    num_inducing_points = num_data * multiplier

    vanilla_deep_gp = build_vanilla_deep_gp(
        data, search_space, num_inducing_points=num_inducing_points
    )

    # correct num_inducing_points
    for layer in vanilla_deep_gp.f_layers:
        npt.assert_equal(layer.q_mu.shape[0], num_inducing_points)

    # correct num_data
    npt.assert_equal(vanilla_deep_gp.num_data, num_data)
    for layer in vanilla_deep_gp.f_layers:
        npt.assert_equal(layer.num_data, num_data)
