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
In this module, we test the *behaviour* of trieste's iducing point selectors.

"""

from __future__ import annotations

from typing import Callable

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, random_seed
from tests.util.models.gpflow.models import mock_data, svgp_model
from tests.util.models.models import fnc_3x_plus_10
from trieste.data import Dataset
from trieste.models.gpflow import SparseVariational
from trieste.models.gpflow.inducing_point_selectors import (
    ConditionalImprovementReduction,
    ConditionalVarianceReduction,
    InducingPointSelector,
    KMeansInducingPointSelector,
    ModelBasedImprovementQualityFunction,
    QualityFunction,
    RandomSubSampleInducingPointSelector,
    UniformInducingPointSelector,
    UnitQualityFunction,
    greedy_inference_dpp,
)
from trieste.space import Box, SearchSpace


@pytest.mark.parametrize(
    "selector",
    [
        UniformInducingPointSelector(Box([0.0], [1.0])),
        RandomSubSampleInducingPointSelector(),
        KMeansInducingPointSelector(),
        ConditionalVarianceReduction(),
        ConditionalImprovementReduction(),
    ],
)
def test_inducing_point_selectors_raise_if_more_than_one_set_of_inducing_points(
    selector: InducingPointSelector[SparseVariational],
) -> None:
    dataset = Dataset(*mock_data())
    svgp = svgp_model(*mock_data())
    model = SparseVariational(svgp)
    inducing_points = [mock_data()[0], mock_data()[0]]
    with pytest.raises(NotImplementedError):
        selector.calculate_inducing_points(inducing_points, model, dataset)


@pytest.mark.parametrize("more_inducing_points_than_data", [True, False])
@pytest.mark.parametrize(
    "selector",
    [
        UniformInducingPointSelector(Box([0.0], [1.0])),
        RandomSubSampleInducingPointSelector(),
        KMeansInducingPointSelector(),
        ConditionalVarianceReduction(),
        ConditionalImprovementReduction(),
    ],
)
def test_inducing_point_selectors_returns_correctly_shaped_inducing_points(
    selector: InducingPointSelector[SparseVariational],
    more_inducing_points_than_data: bool,
) -> None:
    dataset = Dataset(*mock_data())
    svgp = svgp_model(*mock_data())
    model = SparseVariational(svgp)
    if more_inducing_points_than_data:
        inducing_points = tf.concat([mock_data()[0], mock_data()[0]], 0)
    else:
        inducing_points = mock_data()[0]
    new_inducing_points = selector.calculate_inducing_points(inducing_points, model, dataset)
    npt.assert_array_equal(inducing_points.shape, new_inducing_points.shape)


@random_seed
@pytest.mark.parametrize(
    "selector",
    [
        UniformInducingPointSelector(Box([0.0, -1.0], [1.0, 0.0])),
        RandomSubSampleInducingPointSelector(),
        KMeansInducingPointSelector(),
        ConditionalVarianceReduction(),
        ConditionalImprovementReduction(),
    ],
)
def test_inducing_point_selectors_choose_points_still_in_space(
    selector: InducingPointSelector[SparseVariational],
) -> None:
    search_space = Box([0.0, -1.0], [1.0, 0.0])
    X = tf.constant([[0.01, -0.99], [0.99, -0.01]], dtype=tf.float64)
    Y = fnc_3x_plus_10(X)
    dataset = Dataset(X, Y)
    svgp = svgp_model(X, Y)
    model = SparseVariational(svgp)
    inducing_points = search_space.sample(10)
    new_inducing_points = selector.calculate_inducing_points(inducing_points, model, dataset)
    assert tf.reduce_all([point in search_space for point in new_inducing_points])


@random_seed
@pytest.mark.parametrize(
    "selector_name",
    [
        RandomSubSampleInducingPointSelector,
        KMeansInducingPointSelector,
        ConditionalVarianceReduction,
        ConditionalImprovementReduction,
    ],
)
@pytest.mark.parametrize("recalc_every_model_update", [True, False])
def test_inducing_point_selectors_update_correct_number_of_times(
    selector_name: Callable[[SearchSpace, bool], InducingPointSelector[SparseVariational]],
    recalc_every_model_update: bool,
) -> None:
    selector = selector_name(recalc_every_model_update)  # type: ignore
    dataset = Dataset(*mock_data())
    svgp = svgp_model(*mock_data())
    model = SparseVariational(svgp)
    inducing_points = mock_data()[0]
    new_inducing_points_1 = selector.calculate_inducing_points(inducing_points, model, dataset)
    new_inducing_points_2 = selector.calculate_inducing_points(
        new_inducing_points_1, model, dataset
    )
    npt.assert_raises(AssertionError, npt.assert_allclose, inducing_points, new_inducing_points_1)
    npt.assert_raises(AssertionError, npt.assert_allclose, inducing_points, new_inducing_points_2)
    if recalc_every_model_update:
        npt.assert_raises(
            AssertionError, npt.assert_allclose, inducing_points, new_inducing_points_2
        )
    else:
        npt.assert_array_equal(new_inducing_points_1, new_inducing_points_2)


def test_unit_quality_function_returns_correct_scores() -> None:
    search_space = Box([0.0, -1.0], [1.0, 0.0])
    X = search_space.sample(100)
    Y = fnc_3x_plus_10(X)
    dataset = Dataset(X, Y)
    svgp = svgp_model(X, Y)
    model = SparseVariational(svgp)
    quality_scores = UnitQualityFunction()(model, dataset)
    npt.assert_array_equal(
        quality_scores, tf.ones(tf.shape(dataset.query_points)[0], dtype=tf.float64)
    )


@random_seed
def test_improvement_quality_function_returns_approximately_correct_scores() -> None:
    search_space = Box([0.0, -1.0], [1.0, 0.0])
    X = search_space.sample(10)
    Y = fnc_3x_plus_10(X)
    dataset = Dataset(X, Y)
    svgp = svgp_model(X, Y)
    model = SparseVariational(svgp)
    quality_scores = ModelBasedImprovementQualityFunction()(model, dataset)

    samples = model.sample(dataset.query_points, 10000)[:, :, 0]  # [S, N]
    baseline = tf.reduce_max(tf.reduce_mean(samples, 0))
    empirical_scores = tf.maximum(baseline - samples, 0.0)  # [S, N]
    empirical_scores = tf.reduce_mean(empirical_scores, 0)  # [N]

    npt.assert_allclose(quality_scores, empirical_scores, atol=0.01)


def test_greedy_inference_dpp_raises_errors() -> None:
    search_space = Box([0.0, -1.0], [1.0, 0.0])
    X = search_space.sample(100)
    Y = fnc_3x_plus_10(X)
    dataset = Dataset(X, Y)
    svgp = svgp_model(X, Y)
    model = SparseVariational(svgp)
    quality_scores = UnitQualityFunction()(model, dataset)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):  # dataset must be populated
        greedy_inference_dpp(10, svgp.kernel, quality_scores, dataset=None)  # type: ignore
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):  # dataset size must match quality score
        greedy_inference_dpp(10, svgp.kernel, quality_scores[:-1], dataset)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):  # sample must be smaller than dataset size
        greedy_inference_dpp(101, svgp.kernel, quality_scores, dataset)


@pytest.mark.parametrize(
    "quality_function",
    [UnitQualityFunction(), ModelBasedImprovementQualityFunction()],
)
@pytest.mark.parametrize("num_points", [1, 10, 50])
def test_greedy_inference_dpp_returns_correct_number_of_points(
    quality_function: QualityFunction, num_points: int
) -> None:
    search_space = Box([0.0, -1.0], [1.0, 0.0])
    X = search_space.sample(100)
    Y = fnc_3x_plus_10(X)
    dataset = Dataset(X, Y)
    svgp = svgp_model(X, Y)
    model = SparseVariational(svgp)
    quality_scores = quality_function(model, dataset)
    sample = greedy_inference_dpp(num_points, svgp.kernel, quality_scores, dataset)

    assert sample.shape[0] == num_points
    assert sample.shape[1] == search_space.dimension
