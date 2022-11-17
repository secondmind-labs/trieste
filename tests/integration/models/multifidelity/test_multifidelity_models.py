import gpflow
import numpy as np
import numpy.testing as npt
import tensorflow as tf

import trieste
from trieste.data import (
    Dataset,
    add_fidelity_column,
    check_and_extract_fidelity_query_points,
    split_dataset_by_fidelity,
)
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.gpflow.builders import build_gpr, build_multifidelity_autoregressive_models
from trieste.models.gpflow.models import MultifidelityAutoregressive
from trieste.objectives.utils import mk_observer
from trieste.space import Box
from trieste.types import TensorType


def noisy_linear_multifidelity(x: TensorType) -> TensorType:

    x_input, x_fidelity = check_and_extract_fidelity_query_points(x)

    f = 0.5 * ((6.0 * x_input - 2.0) ** 2) * tf.math.sin(12.0 * x_input - 4.0) + 10.0 * (
        x_input - 1.0
    )
    f = f + x_fidelity * (f - 20.0 * (x_input - 1.0))
    noise = tf.random.normal(f.shape, stddev=1e-1, dtype=f.dtype)
    f = tf.where(x_fidelity > 0, f + noise, f)
    return f


def test_multifidelity_autoregressive_results_close() -> None:

    input_dim = 1
    lb = np.zeros(input_dim)
    ub = np.ones(input_dim)
    n_fidelities = 4

    input_search_space = trieste.space.Box(lb, ub)
    n_samples_per_fidelity = [
        2 ** ((n_fidelities - fidelity) + 1) + 3 for fidelity in range(n_fidelities)
    ]

    xs = [tf.linspace(0, 1, samples)[:, None] for samples in n_samples_per_fidelity]
    initial_samples_list = [tf.concat([x, tf.ones_like(x) * i], 1) for i, x in enumerate(xs)]
    initial_sample = tf.concat(initial_samples_list, 0)
    observer = mk_observer(noisy_linear_multifidelity)
    initial_data = observer(initial_sample)

    data = split_dataset_by_fidelity(initial_data, n_fidelities)

    gprs = [
        GaussianProcessRegression(
            build_gpr(
                data[fidelity], input_search_space, likelihood_variance=1e-6, kernel_priors=False
            )
        )
        for fidelity in range(n_fidelities)
    ]

    model = MultifidelityAutoregressive(gprs)

    model.update(initial_data)
    model.optimize(initial_data)

    test_xs = tf.linspace(0, 1, 11)[:, None]
    test_xs_w_fid = add_fidelity_column(test_xs, fidelity=3)

    predictions = model.predict(test_xs_w_fid)[0]
    gt_obs = observer(test_xs_w_fid).observations

    npt.assert_allclose(predictions, gt_obs, rtol=0.20)


def test_multifidelity_autoregressive_gets_expected_rhos() -> None:

    input_dim = 1
    lb = np.zeros(input_dim)
    ub = np.ones(input_dim)
    n_fidelities = 4

    input_search_space = trieste.space.Box(lb, ub)
    n_samples_per_fidelity = [
        2 ** ((n_fidelities - fidelity) + 1) + 3 for fidelity in range(n_fidelities)
    ]

    xs = [tf.linspace(0, 1, samples)[:, None] for samples in n_samples_per_fidelity]
    initial_samples_list = [tf.concat([x, tf.ones_like(x) * i], 1) for i, x in enumerate(xs)]
    initial_sample = tf.concat(initial_samples_list, 0)
    observer = mk_observer(noisy_linear_multifidelity)
    initial_data = observer(initial_sample)

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(initial_data, n_fidelities, input_search_space)
    )

    model.update(initial_data)
    model.optimize(initial_data)

    expected_rho = [1.0] + [(fidelity + 1) / fidelity for fidelity in range(1, n_fidelities)]
    rhos = [float(rho.numpy()) for rho in model.rho]

    npt.assert_allclose(np.array(expected_rho), np.array(rhos), rtol=0.30)


def test_multifidelity_autoregressive_predict_low_fidelity_are_consistent() -> None:

    xs_low = tf.Variable(np.linspace(0, 10, 100), dtype=tf.float64)[:, None]
    xs_high = tf.Variable(np.linspace(0, 10, 10), dtype=tf.float64)[:, None]
    lf_obs = tf.sin(xs_low)
    hf_obs = 2 * tf.sin(xs_high) + tf.random.normal(
        xs_high.shape, mean=0, stddev=1e-1, dtype=tf.float64
    )

    lf_query_points = add_fidelity_column(xs_low, 0)
    hf_query_points = add_fidelity_column(xs_high, 1)

    lf_dataset = Dataset(lf_query_points, lf_obs)
    hf_dataset = Dataset(hf_query_points, hf_obs)

    dataset = lf_dataset + hf_dataset

    search_space = Box([0.0], [10.0])

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(
            dataset, num_fidelities=2, input_search_space=search_space
        )
    )

    model.update(dataset)
    model.optimize(dataset)

    # Add some high fidelity points to check that predict on different fids works
    test_locations_30 = tf.Variable(np.linspace(0, 10, 30), dtype=tf.float64)[:, None]
    lf_test_locations = add_fidelity_column(test_locations_30, 0)

    lf_prediction_mean, lf_prediction_var = model.predict(lf_test_locations)

    (
        lf_prediction_direct_mean,
        lf_prediction_direct_var,
    ) = model.lowest_fidelity_signal_model.predict(test_locations_30)

    npt.assert_array_equal(lf_prediction_mean, lf_prediction_direct_mean)
    npt.assert_array_equal(lf_prediction_var, lf_prediction_direct_var)


def test_multifidelity_autoregressive_predict_lf_are_consistent_with_multiple_fidelities() -> None:

    xs_low = tf.Variable(np.linspace(0, 10, 100), dtype=tf.float64)[:, None]
    xs_high = tf.Variable(np.linspace(0, 10, 10), dtype=tf.float64)[:, None]
    lf_obs = tf.sin(xs_low)
    hf_obs = 2 * tf.sin(xs_high) + tf.random.normal(
        xs_high.shape, mean=0, stddev=1e-1, dtype=tf.float64
    )

    lf_query_points = add_fidelity_column(xs_low, 0)
    hf_query_points = add_fidelity_column(xs_high, 1)

    lf_dataset = Dataset(lf_query_points, lf_obs)
    hf_dataset = Dataset(hf_query_points, hf_obs)

    dataset = lf_dataset + hf_dataset

    search_space = Box([0.0], [10.0])

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(
            dataset, num_fidelities=2, input_search_space=search_space
        )
    )

    model.update(dataset)
    model.optimize(dataset)

    # Add some high fidelity points to check that predict on different fids works
    test_locations_30 = tf.Variable(np.linspace(0, 10, 60), dtype=tf.float64)[:, None]
    lf_test_locations = add_fidelity_column(test_locations_30, 0)
    test_locations_32 = tf.Variable(np.linspace(0, 10, 32), dtype=tf.float64)[:, None]
    hf_test_locations = add_fidelity_column(test_locations_32, 1)

    concat_test_locations = tf.concat([lf_test_locations, hf_test_locations], axis=0)

    concat_prediction_mean, concat_prediction_var = model.predict(concat_test_locations)
    lf_prediction_mean, lf_prediction_var = (
        concat_prediction_mean[:60],
        concat_prediction_var[:60],
    )

    (
        lf_prediction_direct_mean,
        lf_prediction_direct_var,
    ) = model.lowest_fidelity_signal_model.predict(test_locations_30)

    npt.assert_array_equal(lf_prediction_mean, lf_prediction_direct_mean)  # , rtol=1e-7)
    npt.assert_array_equal(lf_prediction_var, lf_prediction_direct_var)  # , rtol=1e-7)


def test_multifidelity_autoregressive_predict_hf_is_consistent_when_rho_zero() -> None:

    xs_low = tf.Variable(np.linspace(0, 10, 100), dtype=tf.float64)[:, None]
    xs_high = tf.Variable(np.linspace(0, 10, 10), dtype=tf.float64)[:, None]
    lf_obs = tf.sin(xs_low)
    hf_obs = 2 * tf.sin(xs_high) + tf.random.normal(
        xs_high.shape, mean=0, stddev=1e-1, dtype=tf.float64
    )

    lf_query_points = add_fidelity_column(xs_low, 0)
    hf_query_points = add_fidelity_column(xs_high, 1)

    lf_dataset = Dataset(lf_query_points, lf_obs)
    hf_dataset = Dataset(hf_query_points, hf_obs)

    dataset = lf_dataset + hf_dataset

    search_space = Box([0.0], [10.0])

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(
            dataset, num_fidelities=2, input_search_space=search_space
        )
    )

    model.update(dataset)
    model.optimize(dataset)

    model.rho[1] = 0.0  # type: ignore

    test_locations = tf.Variable(np.linspace(0, 10, 32), dtype=tf.float64)[:, None]
    hf_test_locations = add_fidelity_column(test_locations, 1)

    hf_prediction = model.predict(hf_test_locations)
    hf_prediction_direct = model.fidelity_residual_models[1].predict(test_locations)

    npt.assert_array_equal(hf_prediction, hf_prediction_direct)


def test_multifidelity_autoregressive_predict_hf_is_consistent_when_lf_is_flat() -> None:

    xs_low = tf.Variable(np.linspace(0, 10, 100), dtype=tf.float64)[:, None]
    xs_high = tf.Variable(np.linspace(0, 10, 10), dtype=tf.float64)[:, None]
    lf_obs = tf.sin(xs_low)
    hf_obs = 2 * tf.sin(xs_high) + tf.random.normal(
        xs_high.shape, mean=0, stddev=1e-1, dtype=tf.float64
    )

    lf_query_points = add_fidelity_column(xs_low, 0)
    hf_query_points = add_fidelity_column(xs_high, 1)

    lf_dataset = Dataset(lf_query_points, lf_obs)
    hf_dataset = Dataset(hf_query_points, hf_obs)

    dataset = lf_dataset + hf_dataset

    search_space = Box([0.0], [10.0])

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(
            dataset, num_fidelities=2, input_search_space=search_space
        )
    )

    model.update(dataset)
    model.optimize(dataset)

    kernel = gpflow.kernels.Matern52(variance=1e-18, lengthscales=[0.2, 0.2])
    gpr = gpflow.models.GPR(dataset.astuple(), kernel, noise_variance=1e-5)

    model.lowest_fidelity_signal_model = GaussianProcessRegression(gpr)

    # Add some low fidelity points to check that predict on different fids works
    test_locations_30 = tf.Variable(np.linspace(0, 10, 30), dtype=tf.float64)[:, None]
    lf_test_locations = add_fidelity_column(test_locations_30, 0)
    test_locations_32 = tf.Variable(np.linspace(0, 10, 32), dtype=tf.float64)[:, None]
    hf_test_locations = add_fidelity_column(test_locations_32, 1)

    concatenated_test_locations = tf.concat([lf_test_locations, hf_test_locations], axis=0)

    concat_prediction, _ = model.predict(concatenated_test_locations)
    hf_prediction = concat_prediction[30:]
    hf_prediction_direct, _ = model.fidelity_residual_models[1].predict(test_locations_32)

    npt.assert_allclose(hf_prediction, hf_prediction_direct)


def test_multifidelity_autoregressive_predict_hf_is_consistent_when_hf_residual_is_flat() -> None:

    xs_low = tf.Variable(np.linspace(0, 10, 100), dtype=tf.float64)[:, None]
    xs_high = tf.Variable(np.linspace(0, 10, 10), dtype=tf.float64)[:, None]
    lf_obs = tf.sin(xs_low)
    hf_obs = 2 * tf.sin(xs_high) + tf.random.normal(
        xs_high.shape, mean=0, stddev=1e-1, dtype=tf.float64
    )

    lf_query_points = add_fidelity_column(xs_low, 0)
    hf_query_points = add_fidelity_column(xs_high, 1)

    lf_dataset = Dataset(lf_query_points, lf_obs)
    hf_dataset = Dataset(hf_query_points, hf_obs)

    dataset = lf_dataset + hf_dataset

    search_space = Box([0.0], [10.0])

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(
            dataset, num_fidelities=2, input_search_space=search_space
        )
    )

    model.update(dataset)
    model.optimize(dataset)

    kernel = gpflow.kernels.Matern52(variance=1e-18, lengthscales=[0.2, 0.2])
    gpr = gpflow.models.GPR(dataset.astuple(), kernel, noise_variance=1e-5)

    model.fidelity_residual_models[1] = GaussianProcessRegression(gpr)  # type: ignore

    test_locations = tf.Variable(np.linspace(0, 10, 32), dtype=tf.float64)[:, None]
    hf_test_locations = add_fidelity_column(test_locations, 1)

    hf_prediction, _ = model.predict(hf_test_locations)
    hf_prediction_direct = (
        model.rho[1] * model.lowest_fidelity_signal_model.predict(test_locations)[0]
    )

    npt.assert_allclose(hf_prediction, hf_prediction_direct)


def test_multifidelity_autoregressive_sample_low_fidelity_are_consistent() -> None:

    xs_low = tf.Variable(np.linspace(0, 10, 100), dtype=tf.float64)[:, None]
    xs_high = tf.Variable(np.linspace(0, 10, 10), dtype=tf.float64)[:, None]
    lf_obs = tf.sin(xs_low)
    hf_obs = 2 * tf.sin(xs_high) + tf.random.normal(
        xs_high.shape, mean=0, stddev=1e-1, dtype=tf.float64
    )

    lf_query_points = add_fidelity_column(xs_low, 0)
    hf_query_points = add_fidelity_column(xs_high, 1)

    lf_dataset = Dataset(lf_query_points, lf_obs)
    hf_dataset = Dataset(hf_query_points, hf_obs)

    dataset = lf_dataset + hf_dataset

    search_space = Box([0.0], [10.0])

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(
            dataset, num_fidelities=2, input_search_space=search_space
        )
    )

    model.update(dataset)
    model.optimize(dataset)

    # Add some high fidelity points to check that predict on different fids works
    test_locations_30 = tf.Variable(np.linspace(0, 10, 30), dtype=tf.float64)[:, None]
    lf_test_locations = add_fidelity_column(test_locations_30, 0)

    lf_samples = model.sample(lf_test_locations, 1_000_000)  # [S, N]

    lf_samples_direct = model.lowest_fidelity_signal_model.sample(test_locations_30, 1_000_000)

    lf_samples_mean = tf.reduce_mean(lf_samples, axis=0)
    lf_samples_var = tf.math.reduce_variance(lf_samples, axis=0)
    lf_samples_direct_mean = tf.reduce_mean(lf_samples_direct, axis=0)
    lf_samples_direct_var = tf.math.reduce_variance(lf_samples_direct, axis=0)

    npt.assert_allclose(lf_samples_mean, lf_samples_direct_mean, atol=1e-5)
    npt.assert_allclose(lf_samples_var, lf_samples_direct_var, atol=1e-5)


def test_multifidelity_autoregressive_sample_lf_are_consistent_with_multiple_fidelities() -> None:

    xs_low = tf.Variable(np.linspace(0, 10, 100), dtype=tf.float64)[:, None]
    xs_high = tf.Variable(np.linspace(0, 10, 10), dtype=tf.float64)[:, None]
    lf_obs = tf.sin(xs_low)
    hf_obs = 2 * tf.sin(xs_high) + tf.random.normal(
        xs_high.shape, mean=0, stddev=1e-1, dtype=tf.float64
    )

    lf_query_points = add_fidelity_column(xs_low, 0)
    hf_query_points = add_fidelity_column(xs_high, 1)

    lf_dataset = Dataset(lf_query_points, lf_obs)
    hf_dataset = Dataset(hf_query_points, hf_obs)

    dataset = lf_dataset + hf_dataset

    search_space = Box([0.0], [10.0])

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(
            dataset, num_fidelities=2, input_search_space=search_space
        )
    )

    model.update(dataset)
    model.optimize(dataset)

    # Add some high fidelity points to check that predict on different fids works
    test_locations_31 = tf.Variable(np.linspace(0, 10, 31), dtype=tf.float64)[:, None]
    lf_test_locations = add_fidelity_column(test_locations_31, 0)
    test_locations_32 = tf.Variable(np.linspace(0, 10, 32), dtype=tf.float64)[:, None]
    hf_test_locations = add_fidelity_column(test_locations_32, 1)

    concat_test_locations = tf.concat([lf_test_locations, hf_test_locations], axis=0)

    concat_samples = model.sample(concat_test_locations, 1_000_000)
    lf_samples = concat_samples[:, :31]

    lf_samples_direct = model.lowest_fidelity_signal_model.sample(test_locations_31, 1_000_000)

    lf_samples_mean = tf.reduce_mean(lf_samples, axis=0)
    lf_samples_var = tf.math.reduce_variance(lf_samples, axis=0)
    lf_samples_direct_mean = tf.reduce_mean(lf_samples_direct, axis=0)
    lf_samples_direct_var = tf.math.reduce_variance(lf_samples_direct, axis=0)

    npt.assert_allclose(lf_samples_mean, lf_samples_direct_mean, atol=1e-5)
    npt.assert_allclose(lf_samples_var, lf_samples_direct_var, atol=1e-5)


def test_multifidelity_autoregressive_sample_hf_is_consistent_when_rho_zero() -> None:

    xs_low = tf.Variable(np.linspace(0, 10, 100), dtype=tf.float64)[:, None]
    xs_high = tf.Variable(np.linspace(0, 10, 10), dtype=tf.float64)[:, None]
    lf_obs = tf.sin(xs_low)
    hf_obs = 2 * tf.sin(xs_high) + tf.random.normal(
        xs_high.shape, mean=0, stddev=1e-1, dtype=tf.float64
    )

    lf_query_points = add_fidelity_column(xs_low, 0)
    hf_query_points = add_fidelity_column(xs_high, 1)

    lf_dataset = Dataset(lf_query_points, lf_obs)
    hf_dataset = Dataset(hf_query_points, hf_obs)

    dataset = lf_dataset + hf_dataset

    search_space = Box([0.0], [10.0])

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(
            dataset, num_fidelities=2, input_search_space=search_space
        )
    )

    model.update(dataset)
    model.optimize(dataset)

    model.rho[1] = 0.0  # type: ignore

    test_locations = tf.Variable(np.linspace(0, 10, 32), dtype=tf.float64)[:, None]
    hf_test_locations = add_fidelity_column(test_locations, 1)

    hf_samples = model.sample(hf_test_locations, 1_000_000)
    hf_samples_direct = model.fidelity_residual_models[1].sample(test_locations, 1_000_000)

    hf_samples_mean = tf.reduce_mean(hf_samples, axis=0)
    hf_samples_var = tf.math.reduce_variance(hf_samples, axis=0)
    hf_samples_direct_mean = tf.reduce_mean(hf_samples_direct, axis=0)
    hf_samples_direct_var = tf.math.reduce_variance(hf_samples_direct, axis=0)

    npt.assert_allclose(hf_samples_mean, hf_samples_direct_mean, atol=1e-3)
    npt.assert_allclose(hf_samples_var, hf_samples_direct_var, atol=1e-3)


def test_multifidelity_autoregressive_sample_hf_is_consistent_when_lf_is_flat() -> None:

    xs_low = tf.Variable(np.linspace(0, 10, 100), dtype=tf.float64)[:, None]
    xs_high = tf.Variable(np.linspace(0, 10, 10), dtype=tf.float64)[:, None]
    lf_obs = tf.sin(xs_low)
    hf_obs = 2 * tf.sin(xs_high) + tf.random.normal(
        xs_high.shape, mean=0, stddev=1e-1, dtype=tf.float64
    )

    lf_query_points = add_fidelity_column(xs_low, 0)
    hf_query_points = add_fidelity_column(xs_high, 1)

    lf_dataset = Dataset(lf_query_points, lf_obs)
    hf_dataset = Dataset(hf_query_points, hf_obs)

    dataset = lf_dataset + hf_dataset

    search_space = Box([0.0], [10.0])

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(
            dataset, num_fidelities=2, input_search_space=search_space
        )
    )

    model.update(dataset)
    model.optimize(dataset)

    kernel = gpflow.kernels.Matern52(variance=1e-18, lengthscales=[0.2, 0.2])
    gpr = gpflow.models.GPR(dataset.astuple(), kernel, noise_variance=1e-5)

    model.lowest_fidelity_signal_model = GaussianProcessRegression(gpr)

    # Add some low fidelity points to check that predict on different fids works
    test_locations_30 = tf.Variable(np.linspace(0, 10, 30), dtype=tf.float64)[:, None]
    lf_test_locations = add_fidelity_column(test_locations_30, 0)
    test_locations_32 = tf.Variable(np.linspace(0, 10, 32), dtype=tf.float64)[:, None]
    hf_test_locations = add_fidelity_column(test_locations_32, 1)

    concatenated_test_locations = tf.concat([lf_test_locations, hf_test_locations], axis=0)

    concat_samples = model.sample(concatenated_test_locations, 1_000_000)
    hf_samples = concat_samples[:, 30:]
    hf_samples_direct = model.fidelity_residual_models[1].sample(test_locations_32, 1_000_000)

    hf_samples_mean = tf.reduce_mean(hf_samples, axis=0)
    hf_samples_var = tf.math.reduce_variance(hf_samples, axis=0)
    hf_samples_direct_mean = tf.reduce_mean(hf_samples_direct, axis=0)
    hf_samples_direct_var = tf.math.reduce_variance(hf_samples_direct, axis=0)

    npt.assert_allclose(hf_samples_mean, hf_samples_direct_mean, atol=1e-3)
    npt.assert_allclose(hf_samples_var, hf_samples_direct_var, atol=1e-3)


def test_multifidelity_autoregressive_sample_hf_is_consistent_when_hf_residual_is_flat() -> None:

    xs_low = tf.Variable(np.linspace(0, 10, 100), dtype=tf.float64)[:, None]
    xs_high = tf.Variable(np.linspace(0, 10, 10), dtype=tf.float64)[:, None]
    lf_obs = tf.sin(xs_low)
    hf_obs = 2 * tf.sin(xs_high) + tf.random.normal(
        xs_high.shape, mean=0, stddev=1e-1, dtype=tf.float64
    )

    lf_query_points = add_fidelity_column(xs_low, 0)
    hf_query_points = add_fidelity_column(xs_high, 1)

    lf_dataset = Dataset(lf_query_points, lf_obs)
    hf_dataset = Dataset(hf_query_points, hf_obs)

    dataset = lf_dataset + hf_dataset

    search_space = Box([0.0], [10.0])

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(
            dataset, num_fidelities=2, input_search_space=search_space
        )
    )

    model.update(dataset)
    model.optimize(dataset)

    kernel = gpflow.kernels.Matern52(variance=1e-18, lengthscales=[0.2, 0.2])
    gpr = gpflow.models.GPR(dataset.astuple(), kernel, noise_variance=1e-5)

    model.fidelity_residual_models[1] = GaussianProcessRegression(gpr)  # type: ignore

    test_locations = tf.Variable(np.linspace(0, 10, 32), dtype=tf.float64)[:, None]
    hf_test_locations = add_fidelity_column(test_locations, 1)

    hf_samples = model.sample(hf_test_locations, 1_000_000)
    hf_samples_direct = model.rho[1] * model.lowest_fidelity_signal_model.sample(
        test_locations, 1_000_000
    )

    hf_samples_mean = tf.reduce_mean(hf_samples, axis=0)
    hf_samples_var = tf.math.reduce_variance(hf_samples, axis=0)
    hf_samples_direct_mean = tf.reduce_mean(hf_samples_direct, axis=0)
    hf_samples_direct_var = tf.math.reduce_variance(hf_samples_direct, axis=0)

    npt.assert_allclose(hf_samples_mean, hf_samples_direct_mean, atol=1e-5, rtol=1e-3)
    npt.assert_allclose(hf_samples_var, hf_samples_direct_var, atol=1e-5, rtol=1e-3)
