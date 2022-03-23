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

import tempfile
import unittest.mock
from collections.abc import Mapping
from itertools import product, zip_longest
from time import sleep
from typing import Optional

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import FixedAcquisitionRule, mk_dataset
from tests.util.models.gpflow.models import PseudoTrainableProbModel, QuadraticMeanAndRBFKernel
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.logging import (
    SummaryFilter,
    default_summary_filter,
    get_current_name_scope,
    get_step_number,
    get_summary_filter,
    get_tensorboard_writer,
    histogram,
    include_summary,
    scalar,
    set_step_number,
    set_summary_filter,
    set_tensorboard_writer,
    step_number,
    tensorboard_writer,
    text,
)
from trieste.models import ProbabilisticModel
from trieste.space import Box, SearchSpace
from trieste.types import TensorType


class _PseudoTrainableQuadratic(QuadraticMeanAndRBFKernel, PseudoTrainableProbModel):
    pass


def test_get_tensorboard_writer_default() -> None:
    assert get_tensorboard_writer() is None


def test_set_get_tensorboard_writer() -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        summary_writer = tf.summary.create_file_writer(tmpdirname)
        set_tensorboard_writer(summary_writer)
        assert get_tensorboard_writer() is summary_writer
        set_tensorboard_writer(None)
        assert get_tensorboard_writer() is None


def test_tensorboard_writer() -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        summary_writer = tf.summary.create_file_writer(tmpdirname)
        assert get_tensorboard_writer() is None
        with tensorboard_writer(summary_writer):
            assert get_tensorboard_writer() is summary_writer
            with tensorboard_writer(None):
                assert get_tensorboard_writer() is None
            assert get_tensorboard_writer() is summary_writer
        assert get_tensorboard_writer() is None


@pytest.mark.parametrize("step", [0, 1, 42])
def test_set_get_step_number(step: int) -> None:
    set_step_number(step)
    assert get_step_number() == step
    set_step_number(0)
    assert get_step_number() == 0


def test_set_step_number_error() -> None:
    with pytest.raises(ValueError):
        set_step_number(-1)


@pytest.mark.parametrize("step", [0, 1, 42])
def test_step_number(step: int) -> None:
    assert get_step_number() == 0
    with step_number(step):
        assert get_step_number() == step
        with step_number(0):
            assert get_step_number() == 0
        assert get_step_number() == step
    assert get_step_number() == 0


@pytest.mark.parametrize("fn", [lambda name: True, lambda name: False, lambda name: "a" in name])
def test_set_get_summary_filter(fn: SummaryFilter) -> None:
    try:
        set_summary_filter(fn)
        assert get_summary_filter() is fn
    finally:
        set_summary_filter(default_summary_filter)


def test_get_current_name_scope() -> None:
    assert get_current_name_scope() == ""
    with tf.name_scope("outer"):
        assert get_current_name_scope() == "outer"
        with tf.name_scope("inner"):
            assert get_current_name_scope() == "outer/inner"
        assert get_current_name_scope() == "outer"
    assert get_current_name_scope() == ""


def test_include_summary() -> None:
    try:
        set_summary_filter(lambda name: "foo" in name)
        assert include_summary("foo")
        assert not include_summary("bar")
        with tf.name_scope("foo"):
            assert include_summary("bar")
    finally:
        set_summary_filter(default_summary_filter)


@unittest.mock.patch("trieste.logging.tf.summary.scalar")
def test_scalar(mocked_summary_scalar: unittest.mock.MagicMock) -> None:
    scalar("this", 1, step=1)
    scalar("_that", 2, step=2)
    with tf.name_scope("foo"):
        scalar("this", 3, step=3)
        scalar("_that", 4, step=4)
    assert len(mocked_summary_scalar.call_args_list) == 2
    for i, j in enumerate([1, 3]):
        assert mocked_summary_scalar.call_args_list[i][0] == ("this", j)
        assert mocked_summary_scalar.call_args_list[i][1] == {"step": j}


@unittest.mock.patch("trieste.logging.tf.summary.histogram")
def test_histogram(mocked_summary_histogram: unittest.mock.MagicMock) -> None:
    histogram("this", tf.constant(1), step=1)
    histogram("_that", tf.constant(2), step=2)
    with tf.name_scope("foo"):
        histogram("this", tf.constant(3), step=3)
        histogram("_that", tf.constant(4), step=4)
    assert len(mocked_summary_histogram.call_args_list) == 2
    for i, j in enumerate([1, 3]):
        assert mocked_summary_histogram.call_args_list[i][0] == ("this", tf.constant(j))
        assert mocked_summary_histogram.call_args_list[i][1] == {"step": j}


@unittest.mock.patch("trieste.logging.tf.summary.text")
def test_text(mocked_summary_histogram: unittest.mock.MagicMock) -> None:
    text("this", "1", step=1)
    text("_that", "2", step=2)
    with tf.name_scope("foo"):
        text("this", "3", step=3)
        text("_that", "4", step=4)
    assert len(mocked_summary_histogram.call_args_list) == 2
    for i, j in enumerate([1, 3]):
        assert mocked_summary_histogram.call_args_list[i][0] == ("this", str(j))
        assert mocked_summary_histogram.call_args_list[i][1] == {"step": j}


@unittest.mock.patch("trieste.models.gpflow.interface.tf.summary.scalar")
def test_tensorboard_logging(mocked_summary_scalar: unittest.mock.MagicMock) -> None:
    mocked_summary_writer = unittest.mock.MagicMock()
    with tensorboard_writer(mocked_summary_writer):
        data, models = {"A": mk_dataset([[0.0]], [[0.0]])}, {"A": _PseudoTrainableQuadratic()}
        steps = 5
        rule = FixedAcquisitionRule([[0.0]])
        BayesianOptimizer(lambda x: {"A": Dataset(x, x ** 2)}, Box([-1], [1])).optimize(
            steps, data, models, rule
        )

    ordered_scalar_names = [
        "A.observation/best_new_observation",
        "A.observation/best_overall",
        "query_points/[0]",
        "wallclock/step",
        "wallclock/query_point_generation",
        "wallclock/model_fitting",
    ]

    for call_arg, (_, scalar_name) in zip_longest(
        mocked_summary_scalar.call_args_list, product(range(steps), ordered_scalar_names)
    ):
        assert call_arg[0][0] == scalar_name
        assert isinstance(call_arg[0][1], float)


@unittest.mock.patch("trieste.models.gpflow.interface.tf.summary.scalar")
@pytest.mark.parametrize("fit_initial_model", [True, False])
def test_wallclock_time_logging(
    mocked_summary_scalar: unittest.mock.MagicMock,
    fit_initial_model: bool,
) -> None:

    model_fit_time = 0.2
    acq_time = 0.1

    class _PseudoTrainableQuadraticWithWaiting(QuadraticMeanAndRBFKernel, PseudoTrainableProbModel):
        def optimize(self, dataset: Dataset) -> None:
            sleep(model_fit_time)

    class _FixedAcquisitionRuleWithWaiting(FixedAcquisitionRule):
        def acquire(
            self,
            search_space: SearchSpace,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> TensorType:
            sleep(acq_time)
            return self._qp

    mocked_summary_writer = unittest.mock.MagicMock()
    with tensorboard_writer(mocked_summary_writer):
        data, models = {"A": mk_dataset([[0.0]], [[0.0]])}, {
            "A": _PseudoTrainableQuadraticWithWaiting()
        }
        steps = 3
        rule = _FixedAcquisitionRuleWithWaiting([[0.0]])
        BayesianOptimizer(lambda x: {"A": Dataset(x, x ** 2)}, Box([-1], [1])).optimize(
            steps, data, models, rule, fit_initial_model=fit_initial_model
        )

    other_scalars = 0

    for i, call_arg in enumerate(mocked_summary_scalar.call_args_list):
        name = call_arg[0][0]
        value = call_arg[0][1]
        step = i // (len(mocked_summary_scalar.call_args_list) / steps)
        if name.startswith("wallclock"):
            assert value > 0  # want positive wallclock times
        if name == "wallclock/step":
            if fit_initial_model and step == 0:
                npt.assert_allclose(value, 2.0 * model_fit_time + acq_time, rtol=0.1)
            else:
                npt.assert_allclose(value, model_fit_time + acq_time, rtol=0.1)
        elif name == "wallclock/query_point_generation":
            npt.assert_allclose(value, acq_time, rtol=0.01)
        elif name == "wallclock/model_fitting":
            if fit_initial_model and step == 0:
                npt.assert_allclose(value, 2.0 * model_fit_time, rtol=0.1)
            else:
                npt.assert_allclose(value, model_fit_time, rtol=0.1)
        else:
            other_scalars += 1

    # check that we processed all the wallclocks we were expecting
    assert len(mocked_summary_scalar.call_args_list) == other_scalars + 3 * steps


@unittest.mock.patch("trieste.models.gpflow.interface.tf.summary.scalar")
def test_tensorboard_logging_ask_tell(mocked_summary_scalar: unittest.mock.MagicMock) -> None:
    mocked_summary_writer = unittest.mock.MagicMock()
    with tensorboard_writer(mocked_summary_writer):
        data, models = {"A": mk_dataset([[0.0]], [[0.0]])}, {"A": _PseudoTrainableQuadratic()}
        rule = FixedAcquisitionRule([[0.0]])
        ask_tell = AskTellOptimizer(Box([-1], [1]), data, models, rule)
        with step_number(3):
            new_point = ask_tell.ask()
            ask_tell.tell({"A": Dataset(new_point, new_point ** 2)})

    ordered_scalar_names = [
        "query_points/[0]",
        "wallclock/query_point_generation",
        "A.observation/best_new_observation",
        "A.observation/best_overall",
        "wallclock/model_fitting",
    ]

    for call_arg, scalar_name in zip_longest(
        mocked_summary_scalar.call_args_list, ordered_scalar_names
    ):
        assert call_arg[0][0] == scalar_name
        assert isinstance(call_arg[0][1], float)
