import numpy as np
import numpy.testing as npt
import tensorflow as tf

import trieste
from trieste.data import (
    check_and_extract_fidelity_query_points,
    convert_query_points_for_fidelity,
    split_dataset_by_fidelity,
)
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.gpflow.builders import build_ar1_models, build_gpr
from trieste.models.gpflow.models import AR1
from trieste.types import TensorType
from trieste.objectives.utils import mk_observer


def noisy_linear_multifidelity(x: TensorType) -> TensorType:

    x_input, x_fidelity = check_and_extract_fidelity_query_points(x)

    f = 0.5 * ((6.0 * x_input - 2.0) ** 2) * tf.math.sin(12.0 * x_input - 4.0) + 10.0 * (
        x_input - 1.0
    )
    f = f + x_fidelity * (f - 20.0 * (x_input - 1.0))
    noise = tf.random.normal(f.shape, stddev=1e-1, dtype=f.dtype)
    f = tf.where(x_fidelity > 0, f + noise, f)
    return f


def test_ar1_results_close() -> None:

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

    model = AR1(gprs)

    print(initial_data.observations.shape)

    model.update(initial_data)
    model.optimize(initial_data)

    test_xs = tf.linspace(0, 1, 11)[:, None]
    test_xs_w_fid = convert_query_points_for_fidelity(test_xs, fidelity=3)

    predictions = model.predict(test_xs_w_fid)[0]
    gt_obs = observer(test_xs_w_fid).observations

    npt.assert_allclose(predictions, gt_obs, rtol=0.20)


def test_ar1_gets_expected_rhos() -> None:

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

    model = AR1(build_ar1_models(initial_data, n_fidelities, input_search_space))

    model.update(initial_data)
    model.optimize(initial_data)

    expected_rho = [1.0] + [(fidelity + 1) / fidelity for fidelity in range(1, n_fidelities)]
    rhos = [float(rho.numpy()) for rho in model.rho]

    print(expected_rho, rhos)

    npt.assert_allclose(np.array(expected_rho), np.array(rhos), rtol=0.30)
