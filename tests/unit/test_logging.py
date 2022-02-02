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
from time import sleep
from typing import Optional

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import FixedAcquisitionRule, mk_dataset
from tests.util.models.gpflow.models import PseudoTrainableProbModel, QuadraticMeanAndRBFKernel
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.logging import (
    get_step_number,
    get_tensorboard_writer,
    set_step_number,
    set_tensorboard_writer,
    step_number,
    tensorboard_writer,
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
        "A.observation.best_overall",
        "A.observation.best_new",
        "Wallclock.total",
        "Wallclock.query_point_generation",
        "Wallclock.model_fitting",
    ]
    N = len(ordered_scalar_names)

    for step in range(steps):  # iterate over saved scalars
        counter = 0
        for scalar_name in ordered_scalar_names:
            assert mocked_summary_scalar.call_args_list[step * N + counter][0][0] == scalar_name
            assert mocked_summary_scalar.call_args_list[step * N + counter][-1]["step"] == step
            assert isinstance(mocked_summary_scalar.call_args_list[step + counter][0][1], float)
            counter += 1


@unittest.mock.patch("trieste.models.gpflow.interface.tf.summary.scalar")
@pytest.mark.parametrize("fit_initial_model", [True, False])
def test_wallclock_time_logging(
    mocked_summary_scalar: unittest.mock.MagicMock,
    fit_initial_model: bool,
) -> None:

    model_fit_time = 0.1
    acq_time = 0.05

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
            """
            :param search_space: Unused.
            :param models: Unused.
            :param datasets: Unused.
            :return: The fixed value specified on initialisation.
            """
            sleep(acq_time)
            return self._qp

    mocked_summary_writer = unittest.mock.MagicMock()
    with tensorboard_writer(mocked_summary_writer):
        data, models = {"A": mk_dataset([[0.0]], [[0.0]])}, {
            "A": _PseudoTrainableQuadraticWithWaiting()
        }
        steps = 2
        rule = _FixedAcquisitionRuleWithWaiting([[0.0]])
        BayesianOptimizer(lambda x: {"A": Dataset(x, x ** 2)}, Box([-1], [1])).optimize(
            steps, data, models, rule, fit_initial_model=fit_initial_model
        )

    for scalar in mocked_summary_scalar.call_args_list:
        if scalar[0][0][:9] == "Wallclock":
            assert scalar[0][1] > 0  # want positive wallclock times

    # check logs of first BO step
    if fit_initial_model:  # logging should count two model fits
        npt.assert_allclose(
            mocked_summary_scalar.call_args_list[2][0][1], 2 * model_fit_time + acq_time, rtol=0.1
        )
        npt.assert_allclose(
            mocked_summary_scalar.call_args_list[4][0][1], 2 * model_fit_time, rtol=0.1
        )
    else:
        npt.assert_allclose(
            mocked_summary_scalar.call_args_list[2][0][1], 1 * model_fit_time + acq_time, rtol=0.1
        )
        npt.assert_allclose(
            mocked_summary_scalar.call_args_list[4][0][1], 1 * model_fit_time, rtol=0.1
        )

    npt.assert_allclose(mocked_summary_scalar.call_args_list[3][0][1], acq_time, rtol=0.01)

    # also check logs of second BO step
    npt.assert_allclose(
        mocked_summary_scalar.call_args_list[7][0][1], 1 * model_fit_time + acq_time, rtol=0.1
    )
    npt.assert_allclose(mocked_summary_scalar.call_args_list[9][0][1], 1 * model_fit_time, rtol=0.1)
    npt.assert_allclose(mocked_summary_scalar.call_args_list[8][0][1], acq_time, rtol=0.01)
