# Copyright 2020 The Trieste Contributors
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
from typing import Any, cast

import pytest
import tensorflow as tf
import numpy.testing as npt

from trieste.acquisition.config import AcquisitionConfig, create_acquisition_rule
from trieste.acquisition.function import ExpectedImprovement, NegativeLowerConfidenceBound
from trieste.acquisition.rule import (
    AcquisitionRule, EfficientGlobalOptimization, OBJECTIVE, ThompsonSampling, TrustRegion,
)
from trieste.datasets import Dataset
from trieste.space import Box

from tests.util.misc import one_dimensional_range, random_seed
from tests.util.model import QuadraticWithUnitVariance


@random_seed(1234)
@pytest.mark.parametrize("config, expected_rule", [
    (("ego", "ei"), EfficientGlobalOptimization(ExpectedImprovement())),
    (("trust_region", {}, "-LCB", [], {"beta": 1.8}), TrustRegion(NegativeLowerConfidenceBound())),
])
def test_create_acquisition_rule(
    config: AcquisitionConfig, expected_rule: AcquisitionRule[Any, Box]
) -> None:
    search_space = one_dimensional_range(0, 1)
    data = {
        OBJECTIVE: Dataset(tf.constant([[0.5], [3.0], [8.3]]), tf.constant([[0.24], [9.1], [70]]))
    }
    models = {OBJECTIVE: QuadraticWithUnitVariance()}
    state = None

    rule = create_acquisition_rule(config)

    assert isinstance(rule, type(expected_rule))

    qps_from_config, state_from_config = expected_rule.acquire(search_space, data, models, state)
    qps, state = cast(AcquisitionRule[Any, Box], rule).acquire(search_space, data, models, state)
    npt.assert_allclose(qps_from_config, qps, rtol=0.03)
    # this works for TrustRegionState because dataclasses define __eq__ and we're using 1D x and y
    assert state_from_config == state


def test_create_acquisition_function_for_thompson_sampling() -> None:
    rule = create_acquisition_rule(("thompson_sampling", [1000], {"num_query_points": 100}))
    assert isinstance(rule, ThompsonSampling)
    # we can't test ThompsonSampling via its acquire method because it changes size so often
    assert rule._num_search_space_samples == 1000
    assert rule._num_query_points == 100
