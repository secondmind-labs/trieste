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
from __future__ import annotations

import math
import unittest
from typing import Any, Callable, List, Tuple, Type
from unittest.mock import MagicMock, patch

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes.exceptions import ShapeMismatchError
from scipy import stats

from tests.unit.models.gpflow.test_interface import _QuadraticPredictor
from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, ShapeLike, quadratic, random_seed
from tests.util.models.gpflow.models import (
    GaussianProcess,
    QuadraticMeanAndRBFKernel,
    QuadraticMeanAndRBFKernelWithSamplers,
    gpr_model,
    quadratic_mean_rbf_kernel_model,
    rbf,
    svgp_model,
    svgp_model_by_type,
)
from trieste.data import Dataset
from trieste.models.gpflow import (
    BatchReparametrizationSampler,
    DecoupledTrajectorySampler,
    GaussianProcessRegression,
    IndependentReparametrizationSampler,
    RandomFourierFeatureTrajectorySampler,
    SparseVariational,
    feature_decomposition_trajectory,
)
from trieste.models.gpflow.sampler import (
    FeatureDecompositionTrajectorySamplerModel,
    qmc_normal_samples,
)
from trieste.models.interfaces import (
    ReparametrizationSampler,
    SupportsGetInducingVariables,
    SupportsPredictJoint,
)
from trieste.objectives import Branin
from trieste.types import TensorType

REPARAMETRIZATION_SAMPLERS: List[Type[ReparametrizationSampler[SupportsPredictJoint]]] = [
    BatchReparametrizationSampler,
    IndependentReparametrizationSampler,
]


DecoupledSamplingModel = Callable[[Dataset], Tuple[int, FeatureDecompositionTrajectorySamplerModel]]


@pytest.fixture(name="sampling_dataset")
def _sampling_dataset() -> Dataset:
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)
    return dataset


@pytest.fixture(
    name="decoupled_sampling_model",
    params=[
        pytest.param(
            lambda dataset: (1, quadratic_mean_rbf_kernel_model(dataset)), id="one_op_custom"
        ),
        # whiten testing is covered in tests/unit/models/gpflow/test_models.py
        pytest.param(
            lambda dataset: (
                1,
                SparseVariational(svgp_model(dataset.query_points, dataset.observations)),
            ),
            id="one_op_svgp",
        ),
        pytest.param(
            lambda dataset: (
                2,
                SparseVariational(
                    svgp_model_by_type(dataset.query_points, "separate+shared", whiten=False)
                ),
            ),
            id="two_op_svgp",
        ),
    ],
)
def _decoupled_sampling_model_fixture(request: Any) -> DecoupledSamplingModel:
    return request.param


@pytest.mark.parametrize(
    "sampler",
    REPARAMETRIZATION_SAMPLERS,
)
def test_reparametrization_sampler_reprs(
    sampler: type[BatchReparametrizationSampler | IndependentReparametrizationSampler],
) -> None:
    assert (
        repr(sampler(20, QuadraticMeanAndRBFKernel()))
        == f"{sampler.__name__}(20, QuadraticMeanAndRBFKernel())"
    )


@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("sample_size", [0, -2])
def test_independent_reparametrization_sampler_raises_for_invalid_sample_size(
    sample_size: int,
    qmc: bool,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        IndependentReparametrizationSampler(sample_size, QuadraticMeanAndRBFKernel(), qmc=qmc)


@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("shape", [[], [1], [2], [2, 3]])
def test_independent_reparametrization_sampler_sample_raises_for_invalid_at_shape(
    shape: ShapeLike,
    qmc: bool,
) -> None:
    sampler = IndependentReparametrizationSampler(1, QuadraticMeanAndRBFKernel(), qmc=qmc)

    with pytest.raises(ShapeMismatchError):
        sampler.sample(tf.zeros(shape))


def _assert_kolmogorov_smirnov_95(
    # fmt: off
    samples: tf.Tensor,  # [..., S]
    distribution: tfp.distributions.Distribution
    # fmt: on
) -> None:
    assert distribution.event_shape == ()
    tf.debugging.assert_shapes([(samples, [..., "S"])])

    sample_size = samples.shape[-1]
    samples_sorted = tf.sort(samples, axis=-1)  # [..., S]
    edf = tf.range(1.0, sample_size + 1, dtype=samples.dtype) / sample_size  # [S]
    expected_cdf = distribution.cdf(samples_sorted)  # [..., S]

    _95_percent_bound = 1.36 / math.sqrt(sample_size)
    assert tf.reduce_max(tf.abs(edf - expected_cdf)) < _95_percent_bound


def _dim_two_gp(mean_shift: tuple[float, float] = (0.0, 0.0)) -> GaussianProcess:
    matern52 = tfp.math.psd_kernels.MaternFiveHalves(
        amplitude=tf.cast(2.3, tf.float64), length_scale=tf.cast(0.5, tf.float64)
    )
    return GaussianProcess(
        [lambda x: mean_shift[0] + Branin.objective(x), lambda x: mean_shift[1] + quadratic(x)],
        [matern52, rbf()],
    )


@random_seed
@unittest.mock.patch(
    "trieste.models.gpflow.sampler.qmc_normal_samples", side_effect=qmc_normal_samples
)
@pytest.mark.parametrize("qmc", [True, False])
def test_independent_reparametrization_sampler_samples_approximate_expected_distribution(
    mocked_qmc: MagicMock, qmc: bool
) -> None:
    sample_size = 1000
    x = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)

    model = _dim_two_gp()
    samples = IndependentReparametrizationSampler(sample_size, model, qmc=qmc).sample(
        x
    )  # [N, S, 1, L]

    assert samples.shape == [len(x), sample_size, 1, 2]

    mean, var = model.predict(tf.squeeze(x, -2))  # [N, L], [N, L]
    _assert_kolmogorov_smirnov_95(
        tf.linalg.matrix_transpose(tf.squeeze(samples, -2)),
        tfp.distributions.Normal(mean[..., None], tf.sqrt(var[..., None])),
    )
    assert mocked_qmc.call_count == qmc


@random_seed
@pytest.mark.parametrize(
    "compiler",
    [
        pytest.param(lambda x: x, id="uncompiled"),
        pytest.param(tf.function, id="tf_function"),
        pytest.param(tf.function(jit_compile=True), id="jit_compile"),
    ],
)
@pytest.mark.parametrize("qmc", [True, False])
def test_independent_reparametrization_sampler_sample_is_continuous(
    compiler: Callable[..., Any], qmc: bool
) -> None:
    sampler = IndependentReparametrizationSampler(100, _dim_two_gp(), qmc=qmc, qmc_skip=False)
    sample = compiler(sampler.sample)
    xs = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_array_less(tf.abs(sample(xs + 1e-20) - sample(xs)), 1e-20)


@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("qmc_skip", [True, False])
def test_independent_reparametrization_sampler_sample_is_repeatable(
    qmc: bool, qmc_skip: bool
) -> None:
    sampler = IndependentReparametrizationSampler(100, _dim_two_gp(), qmc=qmc, qmc_skip=qmc_skip)
    xs = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))


@random_seed
@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("qmc_skip", [True, False])
def test_independent_reparametrization_sampler_samples_are_distinct_for_new_instances(
    qmc: bool,
    qmc_skip: bool,
) -> None:
    sampler1 = IndependentReparametrizationSampler(100, _dim_two_gp(), qmc=qmc, qmc_skip=qmc_skip)
    sampler2 = IndependentReparametrizationSampler(100, _dim_two_gp(), qmc=qmc, qmc_skip=qmc_skip)
    xs = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    if qmc and not qmc_skip:
        npt.assert_raises(
            AssertionError,
            npt.assert_array_less,
            1e-9,
            tf.abs(sampler2.sample(xs) - sampler1.sample(xs)),
        )
    else:
        npt.assert_array_less(1e-9, tf.abs(sampler2.sample(xs) - sampler1.sample(xs)))


@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("qmc_skip", [True, False])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_independent_reparametrization_sampler_dtype(
    qmc: bool, qmc_skip: bool, dtype: tf.DType
) -> None:
    model = QuadraticMeanAndRBFKernel()
    sampler = IndependentReparametrizationSampler(2, model, qmc=qmc, qmc_skip=qmc_skip)
    xs = tf.random.uniform([5, 1, 2], minval=-10.0, maxval=10.0, dtype=dtype)
    samples = sampler.sample(xs)
    assert samples.dtype is dtype


@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("qmc_skip", [True, False])
def test_independent_reparametrization_sampler_reset_sampler(qmc: bool, qmc_skip: bool) -> None:
    sampler = IndependentReparametrizationSampler(100, _dim_two_gp(), qmc=qmc, qmc_skip=qmc_skip)
    assert not sampler._initialized
    xs = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    samples1 = sampler.sample(xs)
    assert sampler._initialized
    sampler.reset_sampler()
    assert not sampler._initialized
    samples2 = sampler.sample(xs)
    assert sampler._initialized
    if qmc and not qmc_skip:
        npt.assert_raises(AssertionError, npt.assert_array_less, 1e-9, tf.abs(samples2 - samples1))
    else:
        npt.assert_array_less(1e-9, tf.abs(samples2 - samples1))


@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_independent_reparametrization_sampler_sample_ensures_positive_variance(
    qmc: bool, dtype: tf.DType
) -> None:
    model = QuadraticMeanAndRBFKernel(kernel_amplitude=tf.constant(0, dtype=dtype))
    sampler = IndependentReparametrizationSampler(100, model, qmc=qmc)
    x = tf.constant([[1.0]], dtype=dtype)
    _, model_var = model.predict(x)
    npt.assert_array_equal(model_var, tf.constant([[0]]))
    variance = tf.math.reduce_variance(sampler.sample(x))  # default jitter
    assert variance > 0
    variance = tf.math.reduce_variance(sampler.sample(x, jitter=-15))  # explicit negative jitter
    assert variance > 0


@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("sample_size", [0, -2])
def test_batch_reparametrization_sampler_raises_for_invalid_sample_size(
    sample_size: int, qmc: bool
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchReparametrizationSampler(sample_size, _dim_two_gp(), qmc=qmc)


@random_seed
@unittest.mock.patch(
    "trieste.models.gpflow.sampler.qmc_normal_samples", side_effect=qmc_normal_samples
)
@pytest.mark.parametrize("qmc", [True, False])
def test_batch_reparametrization_sampler_samples_approximate_mean_and_covariance(
    mocked_qmc: MagicMock, qmc: bool
) -> None:
    model = _dim_two_gp()
    sample_size = 10_000
    leading_dims = [3]
    batch_size = 4
    xs = tf.random.uniform(leading_dims + [batch_size, 2], maxval=1.0, dtype=tf.float64)
    samples = BatchReparametrizationSampler(sample_size, model, qmc=qmc).sample(xs)
    assert mocked_qmc.call_count == qmc
    if qmc:
        assert mocked_qmc.call_args[0][0] == 2 * sample_size  # num_results
        assert mocked_qmc.call_args[0][1] == batch_size  # dim

    assert samples.shape == leading_dims + [sample_size, batch_size, 2]

    samples_mean = tf.reduce_mean(samples, axis=-3)
    samples_covariance = tf.transpose(
        tfp.stats.covariance(samples, sample_axis=-3, event_axis=-2), [0, 3, 1, 2]
    )

    model_mean, model_cov = model.predict_joint(xs)

    npt.assert_allclose(samples_mean, model_mean, rtol=0.02)
    npt.assert_allclose(samples_covariance, model_cov, rtol=0.04)


@pytest.mark.parametrize(
    "compiler",
    [
        pytest.param(lambda x: x, id="uncompiled"),
        pytest.param(tf.function, id="tf_function"),
        pytest.param(tf.function(jit_compile=True), id="jit_compile"),
    ],
)
@pytest.mark.parametrize("qmc", [True, False])
def test_batch_reparametrization_sampler_samples_are_continuous(
    compiler: Callable[..., Any], qmc: bool
) -> None:
    sampler = BatchReparametrizationSampler(100, _dim_two_gp(), qmc=qmc, qmc_skip=False)
    sample = compiler(sampler.sample)
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    npt.assert_array_less(tf.abs(sample(xs + 1e-20) - sample(xs)), 1e-20)


@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("qmc_skip", [True, False])
def test_batch_reparametrization_sampler_samples_are_repeatable(qmc: bool, qmc_skip: bool) -> None:
    sampler = BatchReparametrizationSampler(100, _dim_two_gp(), qmc=qmc, qmc_skip=qmc_skip)
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))


@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("qmc_skip", [True, False])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_batch_reparametrization_sampler_doesnt_cast(
    qmc: bool, qmc_skip: bool, dtype: tf.DType
) -> None:
    sampler = BatchReparametrizationSampler(100, _QuadraticPredictor(), qmc=qmc, qmc_skip=qmc_skip)
    xs = tf.random.uniform([3, 1, 7, 7], dtype=dtype)

    original_tf_cast = tf.cast

    def patched_tf_cast(x: TensorType, dtype: tf.DType) -> TensorType:
        # ensure there are no unnecessary casts from float64 to float32 or vice versa
        if isinstance(x, tf.Tensor) and x.dtype in (tf.float32, tf.float64) and x.dtype != dtype:
            raise ValueError(f"unexpected cast: {x} to {dtype}")
        return original_tf_cast(x, dtype)

    with patch("tensorflow.cast", side_effect=patched_tf_cast):
        samples = sampler.sample(xs)
        assert samples.dtype is dtype
        npt.assert_allclose(samples, sampler.sample(xs))


@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("qmc_skip", [True, False])
def test_batch_reparametrization_sampler_different_batch_sizes(qmc: bool, qmc_skip: bool) -> None:
    sampler = BatchReparametrizationSampler(100, _dim_two_gp(), qmc=qmc, qmc_skip=qmc_skip)
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))
    sampler.reset_sampler()
    xs = tf.random.uniform([3, 5, 10, 2], dtype=tf.float64)
    npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))


@random_seed
@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("qmc_skip", [True, False])
def test_batch_reparametrization_sampler_samples_are_distinct_for_new_instances(
    qmc: bool, qmc_skip: bool
) -> None:
    model = _dim_two_gp()
    sampler1 = BatchReparametrizationSampler(100, model, qmc=qmc, qmc_skip=qmc_skip)
    sampler2 = BatchReparametrizationSampler(100, model, qmc=qmc, qmc_skip=qmc_skip)
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    if qmc and not qmc_skip:
        npt.assert_raises(
            AssertionError,
            npt.assert_array_less,
            1e-9,
            tf.abs(sampler2.sample(xs) - sampler1.sample(xs)),
        )
    else:
        npt.assert_array_less(1e-9, tf.abs(sampler2.sample(xs) - sampler1.sample(xs)))


@pytest.mark.parametrize("at", [tf.constant([0.0]), tf.constant(0.0), tf.ones([0, 1])])
@pytest.mark.parametrize("qmc", [True, False])
def test_batch_reparametrization_sampler_sample_raises_for_invalid_at_shape(
    at: tf.Tensor, qmc: bool
) -> None:
    sampler = BatchReparametrizationSampler(100, QuadraticMeanAndRBFKernel(), qmc=qmc)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(at)


@pytest.mark.parametrize("qmc", [True, False])
def test_batch_reparametrization_sampler_sample_raises_for_negative_jitter(qmc: bool) -> None:
    sampler = BatchReparametrizationSampler(100, QuadraticMeanAndRBFKernel(), qmc=qmc)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.constant([[0.0]]), jitter=-1e-6)


@pytest.mark.parametrize("qmc", [True, False])
def test_batch_reparametrization_sampler_sample_raises_for_inconsistent_batch_size(
    qmc: bool,
) -> None:
    sampler = BatchReparametrizationSampler(100, QuadraticMeanAndRBFKernel(), qmc=qmc)
    sampler.sample(tf.constant([[0.0], [1.0], [2.0]]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.constant([[0.0], [1.0]]))


@pytest.mark.parametrize("qmc", [True, False])
@pytest.mark.parametrize("qmc_skip", [True, False])
def test_batch_reparametrization_sampler_reset_sampler(qmc: bool, qmc_skip: bool) -> None:
    sampler = BatchReparametrizationSampler(
        100, QuadraticMeanAndRBFKernel(), qmc=qmc, qmc_skip=qmc_skip
    )
    assert not sampler._initialized
    xs = tf.constant([[0.0], [1.0], [2.0]])
    samples1 = sampler.sample(xs)
    assert sampler._initialized
    sampler.reset_sampler()
    assert not sampler._initialized
    samples2 = sampler.sample(xs)
    assert sampler._initialized
    if qmc and not qmc_skip:
        npt.assert_raises(AssertionError, npt.assert_array_less, 1e-9, tf.abs(samples2 - samples1))
    else:
        npt.assert_array_less(1e-9, tf.abs(samples2 - samples1))


@pytest.mark.parametrize("num_features", [0, -2])
def test_rff_trajectory_sampler_raises_for_invalid_number_of_features(
    num_features: int,
) -> None:
    dataset = Dataset(
        tf.constant([[-2.0]], dtype=tf.float64), tf.constant([[4.1]], dtype=tf.float64)
    )
    model = quadratic_mean_rbf_kernel_model(dataset)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        RandomFourierFeatureTrajectorySampler(model, num_features=num_features)


def test_rff_trajectory_sampler_raises_for_a_non_gpflow_kernel() -> None:
    dataset = Dataset(tf.constant([[-2.0]]), tf.constant([[4.1]]))
    model = QuadraticMeanAndRBFKernelWithSamplers(dataset=dataset)
    with pytest.raises(AssertionError):
        RandomFourierFeatureTrajectorySampler(model, num_features=100)


@pytest.mark.parametrize("num_evals", [1, 5])
@pytest.mark.parametrize("num_features", [5, 10])
@pytest.mark.parametrize("batch_size", [1])
def test_rff_trajectory_sampler_returns_trajectory_function_with_correct_shapes(
    num_evals: int,
    num_features: int,
    batch_size: int,
) -> None:
    dataset = Dataset(
        tf.constant([[-2.0]], dtype=tf.float64), tf.constant([[4.1]], dtype=tf.float64)
    )
    model = quadratic_mean_rbf_kernel_model(dataset)
    sampler = RandomFourierFeatureTrajectorySampler(model, num_features=num_features)

    trajectory = sampler.get_trajectory()
    xs = tf.linspace(
        [-10.0],
        [10.0],
        num_evals,
    )  # [N, D]
    xs = tf.cast(xs, tf.float64)
    xs_with_dummy_batch_dim = tf.expand_dims(xs, -2)  # [N, 1, D]
    xs_with_full_batch_dim = tf.tile(xs_with_dummy_batch_dim, [1, batch_size, 1])  # [N, B, D]

    tf.debugging.assert_shapes([(trajectory(xs_with_full_batch_dim), [num_evals, batch_size, 1])])
    tf.debugging.assert_shapes(
        [(trajectory._feature_functions(xs), [num_evals, num_features])]  # type: ignore
    )
    assert isinstance(trajectory, feature_decomposition_trajectory)


@random_seed
@pytest.mark.parametrize("batch_size", [1, 5])
def test_rff_trajectory_sampler_returns_deterministic_trajectory(
    batch_size: int,
    sampling_dataset: Dataset,
) -> None:
    model = quadratic_mean_rbf_kernel_model(sampling_dataset)

    sampler = RandomFourierFeatureTrajectorySampler(model, num_features=100)
    trajectory = sampler.get_trajectory()

    xs = sampling_dataset.query_points
    xs = tf.expand_dims(xs, -2)  # [N, 1, D]
    xs = tf.tile(xs, [1, batch_size, 1])  # [N, B, D]
    trajectory_eval_1 = trajectory(xs)
    trajectory_eval_2 = trajectory(xs)

    npt.assert_allclose(trajectory_eval_1, trajectory_eval_2)


def test_rff_trajectory_sampler_returns_same_posterior_from_each_calculation_method(
    sampling_dataset: Dataset,
) -> None:
    model = quadratic_mean_rbf_kernel_model(sampling_dataset)

    sampler = RandomFourierFeatureTrajectorySampler(model, num_features=100)
    sampler.get_trajectory()

    posterior_1 = sampler._prepare_theta_posterior_in_design_space()
    posterior_2 = sampler._prepare_theta_posterior_in_gram_space()

    npt.assert_allclose(posterior_1.loc, posterior_2.loc, rtol=0.02)
    npt.assert_allclose(posterior_1.scale_tril, posterior_2.scale_tril, rtol=0.02)


@random_seed
def test_rff_trajectory_sampler_samples_are_distinct_for_new_instances(
    sampling_dataset: Dataset,
) -> None:
    model = quadratic_mean_rbf_kernel_model(sampling_dataset)

    sampler1 = RandomFourierFeatureTrajectorySampler(model, num_features=100)
    trajectory1 = sampler1.get_trajectory()

    sampler2 = RandomFourierFeatureTrajectorySampler(model, num_features=100)
    trajectory2 = sampler2.get_trajectory()

    xs = sampling_dataset.query_points
    xs = tf.expand_dims(xs, -2)  # [N, 1, d]
    xs = tf.tile(xs, [1, 2, 1])  # [N, 2, D]

    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(trajectory1(xs) - trajectory2(xs)))
    )  # distinct between seperate draws
    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(trajectory1(xs)[:, 0] - trajectory1(xs)[:, 1]))
    )  # distinct for two samples within same draw
    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(trajectory2(xs)[:, 0] - trajectory2(xs)[:, 1]))
    )  # distinct for two samples within same draw


@random_seed
@pytest.mark.parametrize("batch_size", [1, 5])
def test_rff_trajectory_resample_trajectory_provides_new_samples_without_retracing(
    batch_size: int,
    sampling_dataset: Dataset,
) -> None:
    model = quadratic_mean_rbf_kernel_model(sampling_dataset)
    xs = sampling_dataset.query_points
    xs = tf.expand_dims(xs, -2)  # [N, 1, d]
    xs = tf.tile(xs, [1, batch_size, 1])  # [N, B, D]

    sampler = RandomFourierFeatureTrajectorySampler(model, num_features=100)
    trajectory = sampler.get_trajectory()
    evals_1 = trajectory(xs)
    for _ in range(5):
        trajectory = sampler.resample_trajectory(trajectory)
        evals_new = trajectory(xs)
        npt.assert_array_less(
            1e-1, tf.reduce_max(tf.abs(evals_1 - evals_new))
        )  # check all samples are different

    assert trajectory.__call__._get_tracing_count() == 1  # type: ignore


@random_seed
@pytest.mark.parametrize("batch_size", [1, 5])
def test_rff_trajectory_update_trajectory_updates_and_doesnt_retrace(
    batch_size: int,
    sampling_dataset: Dataset,
) -> None:
    model = quadratic_mean_rbf_kernel_model(sampling_dataset)

    x_range = tf.random.uniform([5], 1.0, 2.0)  # sample test locations
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs_predict = tf.reshape(
        tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2)
    )
    xs_predict_with_batching = tf.expand_dims(xs_predict, -2)
    xs_predict_with_batching = tf.tile(xs_predict_with_batching, [1, batch_size, 1])  # [N, B, D]

    trajectory_sampler = RandomFourierFeatureTrajectorySampler(model)
    trajectory = trajectory_sampler.get_trajectory()
    eval_before = trajectory(xs_predict_with_batching)

    for _ in range(3):  # do three updates on new data and see if samples are new
        x_range = tf.random.uniform([5], 1.0, 2.0)
        x_range = tf.cast(x_range, dtype=tf.float64)
        x_train = tf.reshape(
            tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2)
        )

        new_dataset = Dataset(x_train, quadratic(x_train))
        new_lengthscales = 0.5 * model.kernel.lengthscales
        model.update(new_dataset)
        model.kernel.lengthscales.assign(new_lengthscales)  # change params to mimic optimization

        trajectory_updated = trajectory_sampler.update_trajectory(trajectory)
        eval_after = trajectory(xs_predict_with_batching)

        assert trajectory_updated is trajectory  # check update was in place

        npt.assert_allclose(
            trajectory_sampler._feature_functions.kernel.lengthscales, new_lengthscales
        )
        npt.assert_allclose(
            trajectory._feature_functions.kernel.lengthscales, new_lengthscales  # type: ignore
        )
        npt.assert_array_less(
            0.09, tf.reduce_max(tf.abs(eval_before - eval_after))
        )  # two samples should be different

    assert trajectory.__call__._get_tracing_count() == 1  # type: ignore


@pytest.mark.parametrize(
    "sampler_type", [RandomFourierFeatureTrajectorySampler, DecoupledTrajectorySampler]
)
@pytest.mark.parametrize(
    "num_dimensions, active_dims",
    [
        (2, [0]),
        (2, [1]),
        (5, [1, 4]),
        (5, [3, 2, 0]),
    ],
)
@random_seed
def test_trajectory_sampler_respects_active_dims(
    sampler_type: Type[RandomFourierFeatureTrajectorySampler],
    num_dimensions: int,
    active_dims: List[int],
) -> None:
    # Test that the trajectory sampler respects the active_dims setting in a GPflow model.
    num_points = 10
    query_points = tf.random.uniform((num_points, num_dimensions), dtype=tf.float64)
    dataset = Dataset(query_points, quadratic(query_points))

    model = GaussianProcessRegression(gpr_model(dataset.query_points, dataset.observations))
    model.model.kernel = gpflow.kernels.Matern52(active_dims=active_dims)

    num_active_dims = len(active_dims)
    active_dims_mask = tf.scatter_nd(
        tf.expand_dims(active_dims, -1), [True] * num_active_dims, (num_dimensions,)
    )
    x_rnd = tf.random.uniform((num_points, num_dimensions), dtype=tf.float64)
    x_fix = tf.constant(0.5, shape=(num_points, num_dimensions), dtype=tf.float64)
    # We vary values only on the irrelevant dimensions.
    x_test = tf.where(active_dims_mask, x_fix, x_rnd)

    batch_size = 2
    x_test_with_batching = tf.expand_dims(x_test, -2)
    x_test_with_batching = tf.tile(x_test_with_batching, [1, batch_size, 1])  # [N, B, D]
    trajectory_sampler = sampler_type(model)
    trajectory = trajectory_sampler.get_trajectory()

    model_eval = trajectory(x_test_with_batching)
    assert model_eval.shape == (num_points, batch_size, 1)
    # The output should be constant since data only varies on irrelevant dimensions.
    npt.assert_array_almost_equal(
        tf.math.reduce_std(model_eval, axis=0),
        tf.constant(0.0, shape=(batch_size, 1), dtype=tf.float64),
    )


@pytest.mark.parametrize("num_features", [0, -2])
def test_decoupled_trajectory_sampler_raises_for_invalid_number_of_features(
    num_features: int,
) -> None:
    dataset = Dataset(
        tf.constant([[-2.0]], dtype=tf.float64), tf.constant([[4.1]], dtype=tf.float64)
    )
    model = quadratic_mean_rbf_kernel_model(dataset)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        DecoupledTrajectorySampler(model, num_features=num_features)


def test_decoupled_trajectory_sampler_raises_for_a_non_gpflow_kernel() -> None:
    dataset = Dataset(tf.constant([[-2.0]]), tf.constant([[4.1]]))
    model = QuadraticMeanAndRBFKernelWithSamplers(dataset=dataset)

    with pytest.raises(AssertionError):
        DecoupledTrajectorySampler(model, num_features=100)


@pytest.mark.parametrize("num_evals", [10, 100])
@pytest.mark.parametrize("num_features", [5, 50])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_decoupled_trajectory_sampler_returns_trajectory_function_with_correct_shapes(
    num_evals: int,
    num_features: int,
    batch_size: int,
    decoupled_sampling_model: DecoupledSamplingModel,
) -> None:
    dataset = Dataset(
        tf.constant([[-2.0]], dtype=tf.float64), tf.constant([[4.1]], dtype=tf.float64)
    )
    N = len(dataset.query_points)
    L, model = decoupled_sampling_model(dataset)
    sampler = DecoupledTrajectorySampler(model, num_features=num_features)

    trajectory = sampler.get_trajectory()
    xs = tf.linspace([-10.0], [10.0], num_evals)  # [N, D]
    xs = tf.cast(xs, dtype=tf.float64)
    xs_with_dummy_batch_dim = tf.expand_dims(xs, -2)  # [N, 1, D]
    xs_with_full_batch_dim = tf.tile(xs_with_dummy_batch_dim, [1, batch_size, 1])  # [N, B, D]

    tf.debugging.assert_shapes([(trajectory(xs_with_full_batch_dim), [num_evals, batch_size, L])])
    if L > 1:
        tf.debugging.assert_shapes(
            [(trajectory._feature_functions(xs), [L, num_evals, num_features + N])]  # type: ignore
        )
    else:
        tf.debugging.assert_shapes(
            [(trajectory._feature_functions(xs), [num_evals, num_features + N])]  # type: ignore
        )
    assert isinstance(trajectory, feature_decomposition_trajectory)


@random_seed
@pytest.mark.parametrize("batch_size", [1, 5])
def test_decoupled_trajectory_sampler_returns_deterministic_trajectory(
    batch_size: int,
    sampling_dataset: Dataset,
    decoupled_sampling_model: DecoupledSamplingModel,
) -> None:
    _, model = decoupled_sampling_model(sampling_dataset)
    sampler = DecoupledTrajectorySampler(model, num_features=100)
    trajectory = sampler.get_trajectory()

    xs = sampling_dataset.query_points
    xs = tf.expand_dims(xs, -2)  # [N, 1, D]
    xs = tf.tile(xs, [1, batch_size, 1])  # [N, B, D]
    trajectory_eval_1 = trajectory(xs)
    trajectory_eval_2 = trajectory(xs)

    npt.assert_allclose(trajectory_eval_1, trajectory_eval_2)


@random_seed
def test_decoupled_trajectory_sampler_samples_are_distinct_for_new_instances(
    sampling_dataset: Dataset,
) -> None:
    model = quadratic_mean_rbf_kernel_model(sampling_dataset)

    sampler1 = DecoupledTrajectorySampler(model, num_features=100)
    trajectory1 = sampler1.get_trajectory()

    sampler2 = DecoupledTrajectorySampler(model, num_features=100)
    trajectory2 = sampler2.get_trajectory()

    xs = sampling_dataset.query_points
    xs = tf.expand_dims(xs, -2)  # [N, 1, d]
    xs = tf.tile(xs, [1, 2, 1])  # [N, 2, D]
    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(trajectory1(xs) - trajectory2(xs)))
    )  # distinct between sample draws
    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(trajectory1(xs)[:, 0] - trajectory1(xs)[:, 1]))
    )  # distinct between samples within draws
    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(trajectory2(xs)[:, 0] - trajectory2(xs)[:, 1]))
    )  # distinct between samples within draws


@random_seed
@pytest.mark.parametrize("batch_size", [1, 5])
def test_decoupled_trajectory_resample_trajectory_provides_new_samples_without_retracing(
    batch_size: int,
    sampling_dataset: Dataset,
    decoupled_sampling_model: DecoupledSamplingModel,
) -> None:
    _, model = decoupled_sampling_model(sampling_dataset)
    xs = sampling_dataset.query_points
    xs = tf.expand_dims(xs, -2)  # [N, 1, d]
    xs = tf.tile(xs, [1, batch_size, 1])  # [N, B, D]

    sampler = DecoupledTrajectorySampler(model, num_features=100)
    trajectory = sampler.get_trajectory()
    evals_1 = trajectory(xs)
    trace_count_before = trajectory.__call__._get_tracing_count()  # type: ignore
    for _ in range(5):
        trajectory = sampler.resample_trajectory(trajectory)
        evals_new = trajectory(xs)
        npt.assert_array_less(
            1e-1, tf.reduce_max(tf.abs(evals_1 - evals_new))
        )  # check all samples are different

    assert trajectory.__call__._get_tracing_count() == trace_count_before  # type: ignore


@random_seed
@pytest.mark.parametrize("batch_size", [1, 5])
def test_decoupled_trajectory_update_trajectory_updates_and_doesnt_retrace(
    batch_size: int,
    sampling_dataset: Dataset,
    decoupled_sampling_model: DecoupledSamplingModel,
) -> None:
    L, model = decoupled_sampling_model(sampling_dataset)

    x_range = tf.random.uniform([5], 1.0, 2.0)  # sample test locations
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs_predict = tf.reshape(
        tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2)
    )
    xs_predict_with_batching = tf.expand_dims(xs_predict, -2)
    xs_predict_with_batching = tf.tile(xs_predict_with_batching, [1, batch_size, 1])  # [N, B, D]

    trajectory_sampler = DecoupledTrajectorySampler(model)
    trajectory = trajectory_sampler.get_trajectory()
    eval_before = trajectory(xs_predict_with_batching)
    trace_count_before = trajectory.__call__._get_tracing_count()  # type: ignore

    if L > 1:
        # pick the first kernel to check
        _model_lengthscales = model.get_kernel().kernels[0].lengthscales
        _trajectory_sampler_lengthscales = trajectory_sampler._feature_functions.kernel.kernels[
            0
        ].lengthscales
        _trajectory_lengthscales = trajectory._feature_functions.kernel.kernels[  # type: ignore
            0
        ].lengthscales
    else:
        _model_lengthscales = model.get_kernel().lengthscales
        _trajectory_sampler_lengthscales = trajectory_sampler._feature_functions.kernel.lengthscales
        _trajectory_lengthscales = trajectory._feature_functions.kernel.lengthscales  # type: ignore

    for _ in range(3):  # do three updates on new data and see if samples are new
        x_range = tf.random.uniform([5], 1.0, 2.0)
        x_range = tf.cast(x_range, dtype=tf.float64)
        x_train = tf.reshape(
            tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2)
        )

        new_dataset = Dataset(x_train, tf.tile(quadratic(x_train), [1, L]))
        new_lengthscales = 0.5 * _model_lengthscales
        model.update(new_dataset)  # type: ignore
        _model_lengthscales.assign(new_lengthscales)  # change params to mimic optimization

        trajectory_updated = trajectory_sampler.update_trajectory(trajectory)
        eval_after = trajectory(xs_predict_with_batching)

        assert trajectory_updated is trajectory  # check update was in place

        npt.assert_allclose(_trajectory_sampler_lengthscales, new_lengthscales)
        npt.assert_allclose(_trajectory_lengthscales, new_lengthscales)
        npt.assert_array_less(
            0.1, tf.reduce_max(tf.abs(eval_before - eval_after))
        )  # two samples should be different

        # check that inducing points in canonical features closure were updated in place
        if isinstance(model, SupportsGetInducingVariables):
            iv = model.get_inducing_variables()[0]
        else:
            iv = x_train
        npt.assert_array_equal(trajectory_sampler._feature_functions._inducing_points, iv)
        npt.assert_array_equal(trajectory._feature_functions._inducing_points, iv)  # type: ignore

    assert trajectory.__call__._get_tracing_count() == trace_count_before  # type: ignore


@random_seed
@pytest.mark.parametrize("noise_var", [1e-5, 1e-1])
def test_rff_and_decoupled_trajectory_give_similar_results(
    noise_var: float,
    sampling_dataset: Dataset,
) -> None:
    model = quadratic_mean_rbf_kernel_model(sampling_dataset)
    model._noise_variance = tf.constant(noise_var, dtype=tf.float64)

    x_range = tf.linspace(1.4, 1.8, 3)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs_predict = tf.reshape(
        tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2)
    )
    batch_size = 50
    xs_predict_with_batching = tf.expand_dims(xs_predict, -2)
    xs_predict_with_batching = tf.tile(xs_predict_with_batching, [1, batch_size, 1])  # [N, B, D]

    trajectory_sampler_1 = RandomFourierFeatureTrajectorySampler(model)
    trajectory_1 = trajectory_sampler_1.get_trajectory()
    eval_1 = trajectory_1(xs_predict_with_batching)

    trajectory_sampler_2 = DecoupledTrajectorySampler(model)
    trajectory_2 = trajectory_sampler_2.get_trajectory()
    eval_2 = trajectory_2(xs_predict_with_batching)

    npt.assert_allclose(
        tf.reduce_mean(eval_1, 1), tf.reduce_mean(eval_2, 1), rtol=0.01
    )  # means across samples should roughly agree for different samplers
    npt.assert_allclose(
        tf.math.reduce_variance(eval_1, 1), tf.math.reduce_variance(eval_2, 1), rtol=1.0
    )  # variance across samples should (very) roughly agree for different samplers


@pytest.mark.parametrize("n_sample_dim", [0, 2, 5])
@pytest.mark.parametrize("skip", [0, 10_000])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_qmc_samples_return_standard_normal_samples(
    n_sample_dim: int, skip: int, dtype: tf.DType
) -> None:
    n_samples = 10_000

    qmc_samples = qmc_normal_samples(
        num_samples=n_samples, n_sample_dim=n_sample_dim, skip=skip, dtype=dtype
    )
    assert qmc_samples.dtype is dtype
    assert qmc_samples.shape == (n_samples, n_sample_dim)

    # should be multivariate normal with zero correlation
    for i in range(n_sample_dim):
        assert stats.kstest(qmc_samples[:, i], stats.norm.cdf).pvalue > 0.99
        for j in range(n_sample_dim):
            if i != j:
                assert stats.pearsonr(qmc_samples[:, i], qmc_samples[:, j])[0] < 0.005


def test_qmc_samples_skip() -> None:
    samples_1a = qmc_normal_samples(25, 100)
    samples_1b = qmc_normal_samples(25, 100)
    npt.assert_allclose(samples_1a, samples_1b)
    samples_2a = qmc_normal_samples(25, 100, skip=100)
    samples_2b = qmc_normal_samples(25, 100, skip=100)
    npt.assert_allclose(samples_2a, samples_2b)
    npt.assert_raises(AssertionError, npt.assert_allclose, samples_1a, samples_2a)


def test_qmc_samples__num_samples_is_a_tensor() -> None:
    num_samples = 5
    n_sample_dim = 100
    expected_samples = qmc_normal_samples(num_samples, n_sample_dim)
    npt.assert_allclose(
        qmc_normal_samples(tf.constant(num_samples), n_sample_dim), expected_samples
    )
    npt.assert_allclose(
        qmc_normal_samples(tf.constant(num_samples), tf.constant(n_sample_dim)), expected_samples
    )
    npt.assert_allclose(
        qmc_normal_samples(tf.constant(num_samples), n_sample_dim), expected_samples
    )


@pytest.mark.parametrize(
    ("num_samples", "n_sample_dim"),
    (
        [1, 1],
        [0, 1],
        [1, 0],
        [3, 5],
    ),
)
def test_qmc_samples_shapes(num_samples: int, n_sample_dim: int) -> None:
    samples = qmc_normal_samples(num_samples=num_samples, n_sample_dim=n_sample_dim)
    expected_samples_shape = (num_samples, n_sample_dim)
    assert samples.shape == expected_samples_shape


@pytest.mark.parametrize(
    ("num_samples", "n_sample_dim", "skip", "expected_error_type"),
    (
        [-1, 1, 1, tf.errors.InvalidArgumentError],
        [1, -1, 1, tf.errors.InvalidArgumentError],
        [1, 1, -1, tf.errors.InvalidArgumentError],
        [1.5, 1, 1, TypeError],
        [1, 1.5, 1, TypeError],
        [1, 1, 1.5, TypeError],
    ),
)
def test_qmc_samples_shapes__invalid_values(
    num_samples: int, n_sample_dim: int, skip: int, expected_error_type: Any
) -> None:
    with pytest.raises(expected_error_type):
        qmc_normal_samples(num_samples=num_samples, n_sample_dim=n_sample_dim, skip=skip)
