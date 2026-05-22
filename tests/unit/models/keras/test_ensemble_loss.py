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

"""Tests that vectorized ensemble loss matches legacy multi-output aggregation."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
from gpflow.keras import tf_keras

from tests.util.misc import hartmann_6_dataset, random_seed
from tests.util.models.keras.models import trieste_keras_ensemble_model
from trieste.models.keras import DeepEnsemble, build_keras_ensemble, negative_log_likelihood
from trieste.models.keras.utils import compile_metrics_for_ensemble, ensemble_negative_log_likelihood
from trieste.models.optimizer import KerasOptimizer


@random_seed
def test_aggregate_member_loss_scales_with_ensemble_size() -> None:
    """Default Keras reduction on [batch, E] NLL is ~1/E of the legacy summed loss."""
    example_data = hartmann_6_dataset(128)
    ensemble_size = 5
    keras_ensemble = build_keras_ensemble(example_data, ensemble_size, 1, 10, "relu", False)
    deep_ensemble = DeepEnsemble(
        keras_ensemble,
        KerasOptimizer(tf_keras.optimizers.Adam(), {"epochs": 1, "verbose": 0, "batch_size": 32}),
        False,
        False,
    )
    inputs, outputs = deep_ensemble.prepare_dataset(example_data)
    x_name, y_name = keras_ensemble.model.input_names[0], keras_ensemble.model.output_names[0]

    mean_model = tf_keras.models.clone_model(keras_ensemble.model)
    mean_model.set_weights(keras_ensemble.model.get_weights())
    mean_model.compile(optimizer="adam", loss=negative_log_likelihood)
    mean_history = mean_model.fit(inputs, outputs, epochs=1, verbose=0)

    sum_model = tf_keras.models.clone_model(keras_ensemble.model)
    sum_model.set_weights(keras_ensemble.model.get_weights())
    sum_model.compile(optimizer="adam", loss=ensemble_negative_log_likelihood)
    sum_history = sum_model.fit(inputs, outputs, epochs=1, verbose=0)

    npt.assert_allclose(
        sum_history.history["loss"][-1] / mean_history.history["loss"][-1],
        float(ensemble_size),
        rtol=0.05,
    )


@random_seed
def test_deep_ensemble_compile_uses_aggregated_loss() -> None:
    """DeepEnsemble matches manual sum of per-member batch-mean NLL."""
    example_data = hartmann_6_dataset(64)
    ensemble_size = 3
    keras_ensemble = trieste_keras_ensemble_model(example_data, ensemble_size, False)
    model = DeepEnsemble(
        keras_ensemble,
        KerasOptimizer(tf_keras.optimizers.Adam(), {"epochs": 1, "verbose": 0, "batch_size": 16}),
        False,
        False,
    )
    inputs, outputs = model.prepare_dataset(example_data)
    eval_result = model.model.evaluate(inputs, outputs, verbose=0)
    eval_loss = eval_result if isinstance(eval_result, float) else eval_result[0]

    distribution = model.model(inputs)
    manual = float(
        tf.reduce_sum(
            tf.reduce_mean(
                negative_log_likelihood(outputs[keras_ensemble.model.output_names[0]], distribution),
                axis=0,
            )
        )
    )
    npt.assert_allclose(eval_loss, manual, rtol=1e-5)


@random_seed
def test_aggregate_member_losses_wraps_custom_loss() -> None:
    example_data = hartmann_6_dataset(32)
    ensemble_size = 2
    keras_ensemble = build_keras_ensemble(example_data, ensemble_size, 0, 5, "relu", False)
    deep_ensemble = DeepEnsemble(
        keras_ensemble,
        KerasOptimizer(
            tf_keras.optimizers.Adam(),
            {"epochs": 1, "verbose": 0, "batch_size": 16},
            tf_keras.losses.MeanSquaredError(),
        ),
        False,
        False,
    )
    deep_ensemble.optimize(example_data)
    assert deep_ensemble.model.history is not None
    assert deep_ensemble.model.history.history["loss"][-1] >= 0


@random_seed
def test_vectorized_ensemble_fit_with_jit_compile_and_steps_per_execution() -> None:
    """Regression for duplicate ``mse`` metrics under jit_compile + steps_per_execution."""
    example_data = hartmann_6_dataset(64)
    ensemble_size = 3
    keras_ensemble = build_keras_ensemble(example_data, ensemble_size, 1, 10, "relu", False)
    model = DeepEnsemble(
        keras_ensemble,
        KerasOptimizer(
            tf_keras.optimizers.Adam(),
            {"epochs": 2, "verbose": 0, "batch_size": 16},
        ),
        False,
        False,
        compile_args={"jit_compile": True, "steps_per_execution": 10},
    )
    model.optimize(example_data)
    assert model.model.history is not None
    history_keys = model.model.history.history.keys()
    assert "loss" in history_keys
    assert "mse" not in history_keys
    assert len(model.model.history.history["loss"]) == 2


def test_compile_metrics_skipped_when_steps_per_execution_gt_one() -> None:
    """Vectorized ensembles with spe>1 must not register duplicate Keras ``mse`` metrics."""
    assert (
        compile_metrics_for_ensemble(
            1,
            ["mse"],
            ensemble_size=10,
            compile_args={"steps_per_execution": 10},
        )
        is None
    )
