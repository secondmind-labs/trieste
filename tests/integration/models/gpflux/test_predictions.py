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
from trieste.models.gpflux import DeepGaussianProcess, build_vanilla_deep_gp
from trieste.models.optimizer import KerasOptimizer
from trieste.objectives import HARTMANN_6_SEARCH_SPACE


@pytest.fixture(name="depth", params=[2, 3])
def _depth_fixture(request: Any) -> int:
    return request.param


# @pytest.mark.slow
@random_seed
def test_dgp_model_close_to_actuals(depth: int, keras_float: None) -> None:
    dataset_size = 50
    num_inducing = 50

    example_data = hartmann_6_dataset(dataset_size)

    dgp = build_vanilla_deep_gp(
        example_data,
        HARTMANN_6_SEARCH_SPACE,
        depth,
        num_inducing,
        likelihood_variance=1e-5,
        trainable_likelihood=False,
    )
    model = DeepGaussianProcess(dgp)
    model.optimize(example_data)
    predicted_means, _ = model.predict(example_data.query_points)

    np.testing.assert_allclose(predicted_means, example_data.observations, atol=0.2, rtol=0.2)


# @pytest.mark.slow
@random_seed
def test_dgp_model_close_to_simple_implementation(depth: int, keras_float: None) -> None:
    dataset_size = 50
    num_inducing = 50
    batch_size = 50
    epochs = 500
    learning_rate = 0.01

    example_data = hartmann_6_dataset(dataset_size)

    # optimization settings
    def scheduler(epoch: int, lr: float) -> float:
        if epoch == epochs // 2:
            return lr * 0.1
        else:
            return lr

    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
        "callbacks": tf.keras.callbacks.LearningRateScheduler(scheduler),
    }
    optimizer = tf.optimizers.Adam(learning_rate)

    # Trieste implementation
    dgp = build_vanilla_deep_gp(example_data, HARTMANN_6_SEARCH_SPACE, depth, num_inducing)
    trieste_model = DeepGaussianProcess(dgp, KerasOptimizer(optimizer, fit_args))
    trieste_model.optimize(example_data)
    trieste_predicted_means, _ = trieste_model.predict(example_data.query_points)

    # GPflux implementation
    config = Config(
        num_inducing=num_inducing, inner_layer_qsqrt_factor=1e-5, likelihood_noise_variance=1e-2
    )
    gpflux_model = build_constant_input_dim_deep_gp(example_data.query_points, depth, config)
    keras_model = gpflux_model.as_training_model()
    keras_model.compile(optimizer)
    keras_model.fit(
        {"inputs": example_data.query_points, "targets": example_data.observations}, **fit_args
    )
    gpflux_predicted_means, _ = gpflux_model.predict_f(example_data.query_points)

    np.testing.assert_allclose(trieste_predicted_means, gpflux_predicted_means, atol=0.2, rtol=0.2)
