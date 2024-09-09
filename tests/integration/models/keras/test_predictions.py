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

import numpy as np
import pytest
from gpflow.keras import tf_keras

from tests.util.misc import hartmann_6_dataset, random_seed
from trieste.models.keras import DeepEnsemble, build_keras_ensemble
from trieste.models.optimizer import KerasOptimizer


@pytest.mark.slow
@random_seed
def test_neural_network_ensemble_predictions_close_to_actuals() -> None:
    dataset_size = 2000
    example_data = hartmann_6_dataset(dataset_size)

    keras_ensemble = build_keras_ensemble(example_data, 5, 3, 250)
    fit_args = {
        "batch_size": 128,
        "epochs": 1500,
        "callbacks": [
            tf_keras.callbacks.EarlyStopping(
                monitor="loss", patience=100, restore_best_weights=True
            )
        ],
        "verbose": 0,
    }
    model = DeepEnsemble(
        keras_ensemble,
        KerasOptimizer(tf_keras.optimizers.Adam(), fit_args),
    )
    model.optimize(example_data)
    predicted_means, _ = model.predict(example_data.query_points)

    np.testing.assert_allclose(predicted_means, example_data.observations, atol=0.2, rtol=0.2)
