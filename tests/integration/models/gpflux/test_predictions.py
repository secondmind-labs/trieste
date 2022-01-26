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

from typing import Any

import numpy as np
import pytest
import tensorflow as tf
from gpflux.architectures import Config, build_constant_input_dim_deep_gp

from tests.util.misc import hartmann_6_dataset, random_seed
from tests.util.models.gpflux.models import trieste_deep_gaussian_process


@pytest.fixture(name="depth", params=[2, 3])
def _depth_fixture(request: Any) -> int:
    return request.param


@pytest.mark.slow
@random_seed
def test_dgp_model_close_to_actuals(depth: int, keras_float: None) -> None:
    dataset_size = 50
    num_inducing = 50
    batch_size = 50
    epochs = 500

    example_data = hartmann_6_dataset(dataset_size)

    model, _ = trieste_deep_gaussian_process(
        query_points=example_data.query_points,
        depth=depth,
        num_inducing=num_inducing,
        learning_rate=0.01,
        batch_size=batch_size,
        epochs=epochs,
        fix_noise=True,
    )

    model.optimize(example_data)
    predicted_means, _ = model.predict(example_data.query_points)

    np.testing.assert_allclose(predicted_means, example_data.observations, atol=0.2, rtol=0.2)


@pytest.mark.slow
@random_seed
def test_dgp_model_close_to_simple_implementation(depth: int, keras_float: None) -> None:
    dataset_size = 50
    num_inducing = 50
    batch_size = 50
    epochs = 500
    learning_rate = 0.01

    example_data = hartmann_6_dataset(dataset_size)

    # Trieste implementation
    trieste_model, fit_args = trieste_deep_gaussian_process(
        query_points=example_data.query_points,
        depth=depth,
        num_inducing=num_inducing,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
    )
    trieste_model.optimize(example_data)
    trieste_predicted_means, _ = trieste_model.predict(example_data.query_points)

    # GPflux implementation
    config = Config(
        num_inducing=num_inducing, inner_layer_qsqrt_factor=1e-5, likelihood_noise_variance=1e-2
    )

    gpflux_model = build_constant_input_dim_deep_gp(example_data.query_points, depth, config)

    keras_model = gpflux_model.as_training_model()
    keras_model.compile(tf.optimizers.Adam(learning_rate))
    keras_model.fit(
        {"inputs": example_data.query_points, "targets": example_data.observations}, **fit_args
    )

    gpflux_predicted_means, _ = gpflux_model.predict_f(example_data.query_points)

    np.testing.assert_allclose(trieste_predicted_means, gpflux_predicted_means, atol=0.2, rtol=0.2)
