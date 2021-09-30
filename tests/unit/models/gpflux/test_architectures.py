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

import numpy as np
import numpy.testing as npt
import tensorflow as tf
from gpflux.architectures import Config, build_constant_input_dim_deep_gp

from trieste.models.gpflux.architectures import build_vanilla_deep_gp


def test_build_vanilla_deep_gp_returns_correct_model(keras_float: None) -> None:
    num_data = 10
    x = np.arange(num_data).reshape(-1, 1).astype(np.double)

    num_layers = 2
    num_inducing = num_data
    inner_layer_sqrt_factor = 1e-5
    likelihood_noise_variance = 1e-2

    vanilla_deep_gp = build_vanilla_deep_gp(
        x,
        num_layers,
        num_inducing,
        inner_layer_sqrt_factor,
        likelihood_noise_variance,
    )

    config = Config(
        num_inducing,
        inner_layer_sqrt_factor,
        likelihood_noise_variance,
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


def test_build_vanilla_deep_gp_gives_correct_num_inducing(keras_float: None) -> None:
    num_data = 5
    x = np.arange(num_data).reshape(-1, 1).astype(np.double)

    num_layers = 2
    num_inducing = num_data * 2
    inner_layer_sqrt_factor = 1e-5
    likelihood_noise_variance = 1e-2

    vanilla_deep_gp = build_vanilla_deep_gp(
        x,
        num_layers,
        num_inducing,
        inner_layer_sqrt_factor,
        likelihood_noise_variance,
    )

    for layer in vanilla_deep_gp.f_layers:
        npt.assert_equal(layer.q_mu.shape[0], num_inducing)


def test_build_vanilla_deep_gp_gives_correct_num_data(keras_float: None) -> None:
    num_data = 5
    x = np.arange(num_data).reshape(-1, 1).astype(np.double)

    num_layers = 2
    num_inducing = num_data * 2
    inner_layer_sqrt_factor = 1e-5
    likelihood_noise_variance = 1e-2

    vanilla_deep_gp = build_vanilla_deep_gp(
        x,
        num_layers,
        num_inducing,
        inner_layer_sqrt_factor,
        likelihood_noise_variance,
    )

    npt.assert_equal(vanilla_deep_gp.num_data, num_data)

    for layer in vanilla_deep_gp.f_layers:
        npt.assert_equal(layer.num_data, num_data)
