# Copyright (C) Secondmind Ltd 2026 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from gpflow.keras import tf_keras

from trieste.data import Dataset
from trieste.models.keras import DeepEnsemble
from trieste.models.keras.builders import build_keras_ensemble
from trieste.models.optimizer import KerasOptimizer


DATA_DIR = Path(__file__).resolve().parent
PROFILE_DIR = DATA_DIR / "logs" / "profile"


def rmse(y_true: np.ndarray, y: np.ndarray) -> float:
    """
    Root mean squared error between observations and predictive means.

    :param y_true: Ground-truth observations, shape ``[n, 1]``.
    :param y: Predictive means, shape ``[n, 1]``.
    :return: RMSE value.
    """
    return float(np.sqrt(np.mean((y - y_true) ** 2)))


def nlpd(y_true: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    """
    Mean negative log predictive density under a Gaussian predictive distribution.

    :param y_true: Ground-truth observations, shape ``[n, 1]``.
    :param mu: Predictive means, shape ``[n, 1]``.
    :param var: Predictive variances, shape ``[n, 1]``.
    :return: Mean NLPD over test points.
    """
    sigma2 = np.maximum(var, np.finfo(float).eps)
    return float(0.5 * np.mean(np.log(2 * np.pi * sigma2) + (mu - y_true) ** 2 / sigma2))


def build_model(dataset: Dataset) -> DeepEnsemble:
    keras_ensemble = build_keras_ensemble(
        data=dataset,
        ensemble_size=10,
        num_hidden_layers=3,
        units=500,
        activation="tanh",
        independent_normal=False,
    )
    fit_args = {
        "batch_size": 128,
        "epochs": 3,
        "verbose": 1,
    }

    return DeepEnsemble(
        model=keras_ensemble,
        optimizer=KerasOptimizer(
            tf_keras.optimizers.Adam(learning_rate=0.001),
            fit_args,
        ),
        bootstrap=True,
        diversify=True,
        compile_args=dict(
            jit_compile=True,
            steps_per_execution=10,
        ),
    )


if __name__ == "__main__":
    train_data = np.load(DATA_DIR / "train.npz")
    train_dataset = Dataset(
        query_points=train_data["X"],
        observations=train_data["y"].reshape(-1, 1),
    )

    model = build_model(train_dataset)

    shutil.rmtree(PROFILE_DIR, ignore_errors=True)
    PROFILE_DIR.mkdir(parents=True)

    #
    # options = tf.profiler.experimental.ProfilerOptions(
    #     host_tracer_level=2,
    #     python_tracer_level=1,
    #     device_tracer_level=1,
    # )

    # with tf.profiler.experimental.Profile(str(PROFILE_DIR), options=options):
    model.optimize(train_dataset)
    
    test_data = np.load(DATA_DIR / "test.npz")
    mu, var = model.predict_y(test_data["X"])

    y_test = test_data["y"].reshape(-1, 1)
    test_rmse = rmse(y_test, np.asarray(mu))
    test_nlpd = nlpd(y_test, np.asarray(mu), np.asarray(var))
    print("---")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test NLPD: {test_nlpd:.6f}")
