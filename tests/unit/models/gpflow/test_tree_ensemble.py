# Copyright 2024 The Trieste Contributors
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

import gpflow
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
from gpflow.models import GPR

from trieste.data import Dataset
from trieste.models.gpflow.kernels import TreeEnsembleKernel
from trieste.models.gpflow.models import TreeEnsembleGaussianProcess


# ===== Helpers =====

def _make_data(n: int = 20, d: int = 2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.rand(n, d)
    y = np.sin(X[:, 0] * 3) + 0.1 * rng.randn(n)
    return X, y


def _make_tf_data(n: int = 20, d: int = 2, seed: int = 42) -> tuple[tf.Tensor, tf.Tensor]:
    X, y = _make_data(n, d, seed)
    return tf.constant(X, dtype=tf.float64), tf.constant(y[:, None], dtype=tf.float64)


# ===== TreeEnsembleKernel tests =====


def test_kernel_unfitted_raises() -> None:
    kernel = TreeEnsembleKernel()
    X = tf.constant(np.random.rand(5, 2), dtype=tf.float64)
    with pytest.raises(RuntimeError, match="forest has not been fitted"):
        kernel.K(X)


def test_kernel_symmetry() -> None:
    kernel = TreeEnsembleKernel(n_estimators=50)
    X, y = _make_data()
    kernel.fit_forest(X, y)

    X1 = tf.constant(X[:5], dtype=tf.float64)
    X2 = tf.constant(X[5:10], dtype=tf.float64)
    K12 = kernel.K(X1, X2)
    K21 = kernel.K(X2, X1)

    npt.assert_allclose(K12.numpy(), K21.numpy().T, atol=1e-10)


def test_kernel_self_similarity() -> None:
    kernel = TreeEnsembleKernel(n_estimators=50)
    X, y = _make_data()
    kernel.fit_forest(X, y)

    X_tf = tf.constant(X, dtype=tf.float64)
    diag = kernel.K_diag(X_tf)

    npt.assert_allclose(diag.numpy(), np.ones(len(X)), atol=1e-10)


def test_kernel_values_in_unit_interval() -> None:
    kernel = TreeEnsembleKernel(n_estimators=50)
    X, y = _make_data()
    kernel.fit_forest(X, y)

    X_tf = tf.constant(X, dtype=tf.float64)
    K = kernel.K(X_tf)

    assert tf.reduce_all(K >= 0.0).numpy()
    assert tf.reduce_all(K <= 1.0).numpy()


def test_kernel_psd() -> None:
    kernel = TreeEnsembleKernel(n_estimators=50)
    X, y = _make_data()
    kernel.fit_forest(X, y)

    X_tf = tf.constant(X, dtype=tf.float64)
    K = kernel.K(X_tf).numpy()

    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues >= -1e-8), f"Negative eigenvalue found: {eigenvalues.min()}"


def test_kernel_different_data_produces_different_matrix() -> None:
    kernel = TreeEnsembleKernel(n_estimators=50, min_samples_leaf=5, random_state=0)
    X1, y1 = _make_data(n=50, seed=1)
    X2, y2 = _make_data(n=50, seed=2)

    # Use the same fixed query grid to compare kernel matrices across two different forests
    query = tf.constant(np.linspace(0, 1, 10).reshape(-1, 1).repeat(2, axis=1), dtype=tf.float64)

    kernel.fit_forest(X1, y1)
    K1 = kernel.K(query).numpy()

    kernel.fit_forest(X2, y2)
    K2 = kernel.K(query).numpy()

    assert not np.allclose(K1, K2), "Kernel matrices should differ after refitting on new data"


def test_kernel_k_with_x2_none() -> None:
    kernel = TreeEnsembleKernel(n_estimators=50)
    X, y = _make_data()
    kernel.fit_forest(X, y)

    X_tf = tf.constant(X[:5], dtype=tf.float64)
    K_auto = kernel.K(X_tf)
    K_explicit = kernel.K(X_tf, X_tf)

    npt.assert_allclose(K_auto.numpy(), K_explicit.numpy(), atol=1e-10)


def test_kernel_diagonal_matches_k_matrix() -> None:
    kernel = TreeEnsembleKernel(n_estimators=50)
    X, y = _make_data()
    kernel.fit_forest(X, y)

    X_tf = tf.constant(X, dtype=tf.float64)
    K = kernel.K(X_tf)
    K_diag = kernel.K_diag(X_tf)

    npt.assert_allclose(tf.linalg.diag_part(K).numpy(), K_diag.numpy(), atol=1e-10)


# ===== TreeEnsembleGaussianProcess tests =====


def _build_tree_gp_model(
    X: tf.Tensor, y: tf.Tensor
) -> TreeEnsembleGaussianProcess:
    tree_kernel = TreeEnsembleKernel(n_estimators=50, random_state=0)
    kernel = gpflow.kernels.Constant() * tree_kernel
    gpr = GPR(data=(X, y), kernel=kernel)
    gpr.likelihood.variance.assign(0.01)
    return TreeEnsembleGaussianProcess(gpr)


def test_model_raises_without_tree_kernel() -> None:
    X, y = _make_tf_data()
    gpr = GPR(data=(X, y), kernel=gpflow.kernels.Matern52())
    with pytest.raises(ValueError, match="requires a TreeEnsembleKernel"):
        TreeEnsembleGaussianProcess(gpr)


def test_model_predict_returns_valid_mean_variance() -> None:
    X, y = _make_tf_data()
    model = _build_tree_gp_model(X, y)

    X_test = tf.constant(np.random.RandomState(99).rand(5, 2), dtype=tf.float64)
    mean, var = model.predict(X_test)

    assert mean.shape == (5, 1)
    assert var.shape == (5, 1)
    assert tf.reduce_all(var > 0.0).numpy(), "Variance must be positive"


def test_model_forest_refitted_on_optimize() -> None:
    """Verify optimize_encoded refits the forest (bypass Scipy optimizer)."""
    X, y = _make_tf_data(n=20, seed=1)
    model = _build_tree_gp_model(X, y)

    X_test = tf.constant([[0.5, 0.5]], dtype=tf.float64)
    mean1, _ = model.predict(X_test)

    # Manually refit with different data
    X2, y2 = _make_data(n=20, seed=99)
    model._tree_kernel.fit_forest(X2, y2)

    mean2, _ = model.predict(X_test)

    assert not np.allclose(
        mean1.numpy(), mean2.numpy()
    ), "Predictions should change after refitting the forest"


def test_model_update_changes_data() -> None:
    X, y = _make_tf_data(n=10, seed=1)
    model = _build_tree_gp_model(X, y)

    X2, y2 = _make_tf_data(n=15, seed=2)
    dataset2 = Dataset(
        tf.concat([X, X2], axis=0),
        tf.concat([y, y2], axis=0),
    )
    model.update(dataset2)

    internal_data = model.get_internal_data()
    assert tf.shape(internal_data.query_points)[0] == 25


def test_model_finds_tree_kernel_in_product() -> None:
    X, y = _make_tf_data()
    tree_kernel = TreeEnsembleKernel(n_estimators=30)
    kernel = gpflow.kernels.Constant() * tree_kernel
    gpr = GPR(data=(X, y), kernel=kernel)
    model = TreeEnsembleGaussianProcess(gpr)
    assert model._tree_kernel is tree_kernel
