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

from math import pi

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import random_seed
from tests.util.models.gpflux.models import two_layer_dgp_model
from trieste.acquisition.rule import AcquisitionRule, DiscreteThompsonSampling
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.models.gpflux import GPfluxModelConfig
from trieste.objectives import MICHALEWICZ_2_MINIMIZER, MICHALEWICZ_2_MINIMUM, michalewicz
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box

tf.keras.backend.set_floatx("float64")


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        (5, DiscreteThompsonSampling(1000, 50)),
    ],
)
def test_two_layer_dgp_optimizer_finds_minima_of_michalewicz_function(
    num_steps: int, acquisition_rule: AcquisitionRule
) -> None:

    search_space = Box([0, 0], [pi, pi])

    def build_model(data: Dataset) -> GPfluxModelConfig:
        epochs = int(2e3)
        batch_size = 100

        dgp = two_layer_dgp_model(data.query_points)

        def scheduler(epoch, lr):
            if epoch == epochs // 2:
                return lr * 0.1
            else:
                return lr

        optimizer = tf.optimizers.Adam(0.01)
        fit_args = {
            "batch_size": batch_size,
            "epochs": epochs,
            "verbose": 0,
            "callbacks": tf.keras.callbacks.LearningRateScheduler(scheduler),
        }

        config = GPfluxModelConfig(
            **{
                "model": dgp,
                "model_args": {
                    "fit_args": fit_args,
                },
                "optimizer": optimizer,
            }
        )

        return config

    initial_query_points = search_space.sample(50)
    observer = mk_observer(michalewicz, OBJECTIVE)
    initial_data = observer(initial_query_points)
    model = build_model(initial_data[OBJECTIVE])
    dataset = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, {OBJECTIVE: model}, acquisition_rule, track_state=False)
        .try_get_final_dataset()
    )
    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]
    relative_minimizer_err = tf.abs((best_x - MICHALEWICZ_2_MINIMIZER) / MICHALEWICZ_2_MINIMIZER)

    assert tf.reduce_all(relative_minimizer_err < 0.03, axis=-1)
    npt.assert_allclose(best_y, MICHALEWICZ_2_MINIMUM, rtol=0.03)
