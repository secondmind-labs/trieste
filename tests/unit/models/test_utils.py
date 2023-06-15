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

import unittest.mock
from typing import Tuple

import gpflow
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.models.models import fnc_3x_plus_10
from trieste.data import Dataset
from trieste.models import TrainableProbabilisticModel
from trieste.models.utils import (
    write_summary_data_based_metrics,
    write_summary_kernel_parameters,
    write_summary_likelihood_parameters,
)
from trieste.types import TensorType


@unittest.mock.patch("trieste.models.gpflow.interface.tf.summary.scalar")
@pytest.mark.parametrize(
    "kernel, names, values",
    [
        pytest.param(
            gpflow.kernels.Matern32(),
            ["kernel.Matern32.variance", "kernel.Matern32.lengthscales"],
            [1, 1],
            id="Matern32, Default",
        ),
        pytest.param(
            gpflow.kernels.Matern52(variance=2.0, lengthscales=[0.2, 0.2]),
            [
                "kernel.Matern52.variance",
                "kernel.Matern52.lengthscales[0]",
                "kernel.Matern52.lengthscales[1]",
            ],
            [2, 0.2, 0.2],
            id="Matern52, ARD",
        ),
        pytest.param(
            gpflow.kernels.Matern12() * gpflow.kernels.Linear(),
            [
                "kernel.Product.kernels[0].variance",
                "kernel.Product.kernels[0].lengthscales",
                "kernel.Product.kernels[1].variance",
            ],
            [1, 1, 1],
            id="product kernel",
        ),
    ],
)
def test_write_summary_kernel_parameters(
    mocked_summary_scalar: unittest.mock.MagicMock,
    kernel: gpflow.kernels.Kernel,
    names: list[str],
    values: list[float],
) -> None:
    write_summary_kernel_parameters(kernel)

    assert mocked_summary_scalar.call_count == len(names)
    for i, (n, v) in enumerate(zip(names, values)):
        assert mocked_summary_scalar.call_args_list[i][0][0] == n
        assert mocked_summary_scalar.call_args_list[i][0][1].numpy() == v


@unittest.mock.patch("trieste.models.gpflow.interface.tf.summary.scalar")
@pytest.mark.parametrize(
    "likelihood, names, values",
    [
        pytest.param(
            gpflow.likelihoods.Gaussian(),
            ["likelihood.Gaussian.variance"],
            [1],
            id="Gaussian, Default",
        ),
        pytest.param(
            gpflow.likelihoods.Gaussian(scale=0.2),
            ["likelihood.Gaussian.scale"],
            [0.2],
            id="Gaussian, scale",
        ),
        pytest.param(
            gpflow.likelihoods.Gaussian(scale=gpflow.functions.Polynomial(degree=2)),
            ["likelihood.Gaussian.scale.w"],
            [[1, 0, 0]],
            id="Gaussian, polynomial",
        ),
        pytest.param(
            gpflow.likelihoods.Gaussian(
                variance=gpflow.functions.SwitchedFunction(
                    [
                        gpflow.functions.Constant(1.0),
                        gpflow.functions.Constant(1.0),
                    ]
                )
            ),
            [
                "likelihood.Gaussian.variance.functions[0].c",
                "likelihood.Gaussian.variance.functions[1].c",
            ],
            [1, 1],
            id="Gaussian, grouped noise variance",
        ),
        pytest.param(
            gpflow.likelihoods.Beta(),
            ["likelihood.Beta.scale"],
            [1],
            id="Beta, default",
        ),
        pytest.param(
            gpflow.likelihoods.HeteroskedasticTFPConditional(
                distribution_class=tfp.distributions.Normal,
                scale_transform=tfp.bijectors.Exp(),
            ),
            [],
            [],
            id="HeteroskedasticTFPConditional",
        ),
    ],
)
def test_write_summary_likelihood_parameters(
    mocked_summary_scalar: unittest.mock.MagicMock,
    likelihood: gpflow.likelihoods.Likelihood,
    names: list[str],
    values: list[float],
) -> None:
    write_summary_likelihood_parameters(likelihood)

    assert mocked_summary_scalar.call_count == len(names)
    for i, (n, v) in enumerate(zip(names, values)):
        assert mocked_summary_scalar.call_args_list[i][0][0] == n
        assert tf.reduce_all(np.isclose(mocked_summary_scalar.call_args_list[i][0][1].numpy(), v))


@unittest.mock.patch("trieste.logging.tf.summary.histogram")
@unittest.mock.patch("trieste.logging.tf.summary.scalar")
@pytest.mark.parametrize("prefix", ["", "dummy_"])
def test_write_summary_data_based_metrics(
    mocked_summary_scalar: unittest.mock.MagicMock,
    mocked_summary_histogram: unittest.mock.MagicMock,
    prefix: str,
) -> None:
    x = tf.constant(np.arange(1, 5).reshape(-1, 1), dtype=gpflow.default_float())  # shape: [4, 1]
    y = fnc_3x_plus_10(x)
    dataset = Dataset(x, y)

    def _mocked_predict(query_points: TensorType) -> Tuple[TensorType, TensorType]:
        return (
            y,
            tf.math.abs(y),
        )

    mock_model: TrainableProbabilisticModel = unittest.mock.MagicMock(
        spec=TrainableProbabilisticModel
    )
    mock_model.predict = _mocked_predict  # type: ignore

    write_summary_data_based_metrics(dataset=dataset, model=mock_model, prefix=prefix)

    scalar_names_values = [
        (f"{prefix}accuracy/predict_mean__mean", tf.reduce_mean(y)),
        (f"{prefix}accuracy/predict_variance__mean", tf.reduce_mean(tf.math.abs(y))),
        (f"{prefix}accuracy/observations_mean", tf.reduce_mean(y)),
        (f"{prefix}accuracy/observations_variance", tf.math.reduce_variance(y)),
        (f"{prefix}accuracy/root_mean_square_error", 0.0),
        (f"{prefix}accuracy/mean_absolute_error", 0.0),
        (f"{prefix}accuracy/z_residuals_std", 0.0),
        (
            f"{prefix}accuracy/root_mean_variance_error",
            tf.math.sqrt(tf.reduce_mean(tf.math.abs(y) ** 2)),
        ),
    ]
    assert mocked_summary_scalar.call_count == len(scalar_names_values)
    for i, (n, v) in enumerate(scalar_names_values):
        assert mocked_summary_scalar.call_args_list[i][0][0] == n
        assert mocked_summary_scalar.call_args_list[i][0][1].numpy() == v

    histogram_names_values = [
        (f"{prefix}accuracy/predict_mean", y),
        (f"{prefix}accuracy/predict_variance", tf.math.abs(y)),
        (f"{prefix}accuracy/observations", y),
        (f"{prefix}accuracy/absolute_error", y - y),
        (f"{prefix}accuracy/z_residuals", y - y),
        (f"{prefix}accuracy/variance_error", tf.math.abs(y)),
    ]
    assert mocked_summary_histogram.call_count == len(histogram_names_values)
    for i, (n, v) in enumerate(histogram_names_values):
        assert mocked_summary_histogram.call_args_list[i][0][0] == n
        assert tf.reduce_all(mocked_summary_histogram.call_args_list[i][0][1] == v)
