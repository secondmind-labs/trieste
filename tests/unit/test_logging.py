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
from itertools import zip_longest
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
from trieste.types import Tag, TensorType


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
        scalar("this", lambda: 3, step=3)
        scalar("_that", lambda: 4, step=4)
        scalar("broken", lambda: 1 / 0, step=5)
    assert len(mocked_summary_scalar.call_args_list) == 2
    for i, j in enumerate([1, 3]):
        assert mocked_summary_scalar.call_args_list[i][0] == ("this", j)
        assert mocked_summary_scalar.call_args_list[i][1] == {"step": j}


@unittest.mock.patch("trieste.logging.tf.summary.histogram")
def test_histogram(mocked_summary_histogram: unittest.mock.MagicMock) -> None:
    histogram("this", tf.constant(1), step=1)
    histogram("_that", tf.constant(2), step=2)
    with tf.name_scope("foo"):
        histogram("this", lambda: tf.constant(3), step=3)
        histogram("_that", lambda: tf.constant(4), step=4)
        histogram("broken", lambda: tf.constant(1 / 0), step=5)
    assert len(mocked_summary_histogram.call_args_list) == 2
    for i, j in enumerate([1, 3]):
        assert mocked_summary_histogram.call_args_list[i][0] == ("this", tf.constant(j))
        assert mocked_summary_histogram.call_args_list[i][1] == {"step": j}


@unittest.mock.patch("trieste.logging.tf.summary.text")
def test_text(mocked_summary_histogram: unittest.mock.MagicMock) -> None:
    text("this", "1", step=1)
    text("_that", "2", step=2)
    with tf.name_scope("foo"):
        text("this", lambda: "3", step=3)
        text("_that", lambda: "4", step=4)
        text("broken", lambda: f"{1/0}", step=5)
    assert len(mocked_summary_histogram.call_args_list) == 2
    for i, j in enumerate([1, 3]):
        assert mocked_summary_histogram.call_args_list[i][0] == ("this", str(j))
        assert mocked_summary_histogram.call_args_list[i][1] == {"step": j}


@unittest.mock.patch("trieste.models.gpflow.interface.tf.summary.scalar")
def test_tensorboard_logging(mocked_summary_scalar: unittest.mock.MagicMock) -> None:
    mocked_summary_writer = unittest.mock.MagicMock()
    with tensorboard_writer(mocked_summary_writer):
        tag: Tag = "A"
        data, models = {tag: mk_dataset([[0.0]], [[0.0]])}, {tag: _PseudoTrainableQuadratic()}
        steps = 5
        rule = FixedAcquisitionRule([[0.0]])
        BayesianOptimizer(lambda x: {tag: Dataset(x, x**2)}, Box([-1], [1])).optimize(
            steps, data, models, rule
        )

    ordered_scalar_names = [
        "A.observation/best_new_observation",
        "A.observation/best_overall",
        "wallclock/model_fitting",
        "query_point/[0]",
        "wallclock/query_point_generation",
        "wallclock/step",
    ]
    for call_arg, scalar_name in zip_longest(
        mocked_summary_scalar.call_args_list,
        ["wallclock/model_fitting"] + steps * ordered_scalar_names,
    ):
        assert call_arg[0][0] == scalar_name
        assert isinstance(call_arg[0][1], float)


@unittest.mock.patch("trieste.models.gpflow.interface.tf.summary.scalar")
@pytest.mark.parametrize("fit_model", ["all", "all_but_init", "never"])
def test_wallclock_time_logging(
    mocked_summary_scalar: unittest.mock.MagicMock,
    fit_model: str,
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
            models: Mapping[Tag, ProbabilisticModel],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
        ) -> TensorType:
            sleep(acq_time)
            return self._qp

    mocked_summary_writer = unittest.mock.MagicMock()
    with tensorboard_writer(mocked_summary_writer):
        tag: Tag = "A"
        data, models = {tag: mk_dataset([[0.0]], [[0.0]])}, {
            tag: _PseudoTrainableQuadraticWithWaiting()
        }
        steps = 3
        rule = _FixedAcquisitionRuleWithWaiting([[0.0]])
        BayesianOptimizer(lambda x: {tag: Dataset(x, x**2)}, Box([-1], [1])).optimize(
            steps,
            data,
            models,
            rule,
            fit_model=fit_model in ["all", "all_but_init"],
            fit_initial_model=fit_model in ["all"],
        )

    other_scalars = 0

    for i, call_arg in enumerate(mocked_summary_scalar.call_args_list):
        name = call_arg[0][0]
        value = call_arg[0][1]
        if fit_model == "all" and i == 0:
            assert name == "wallclock/model_fitting"
        if name.startswith("wallclock"):
            assert value > 0  # want positive wallclock times
        if name == "wallclock/query_point_generation":
            npt.assert_allclose(value, acq_time, rtol=0.01)
        elif name == "wallclock/step":
            total_time = acq_time if fit_model == "never" else model_fit_time + acq_time
            npt.assert_allclose(value, total_time, rtol=0.1)
        elif name == "wallclock/model_fitting":
            model_time = 0.0 if fit_model == "never" else model_fit_time
            npt.assert_allclose(value, model_time, atol=0.01)
        else:
            other_scalars += 1

    # check that we processed all the wallclocks we were expecting
    total_wallclocks = other_scalars + 3 * steps
    if fit_model == "all":
        total_wallclocks += 1
    assert len(mocked_summary_scalar.call_args_list) == total_wallclocks


@unittest.mock.patch("trieste.models.gpflow.interface.tf.summary.scalar")
def test_tensorboard_logging_ask_tell(mocked_summary_scalar: unittest.mock.MagicMock) -> None:
    mocked_summary_writer = unittest.mock.MagicMock()
    with tensorboard_writer(mocked_summary_writer):
        tag: Tag = "A"
        data, models = {tag: mk_dataset([[0.0]], [[0.0]])}, {tag: _PseudoTrainableQuadratic()}
        rule = FixedAcquisitionRule([[0.0]])
        ask_tell = AskTellOptimizer(Box([-1], [1]), data, models, rule)
        with step_number(3):
            new_point = ask_tell.ask()
            ask_tell.tell({tag: Dataset(new_point, new_point**2)})

    ordered_scalar_names = [
        "wallclock/model_fitting",
        "query_point/[0]",
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
