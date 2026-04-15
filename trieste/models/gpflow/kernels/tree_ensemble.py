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
"""Tree-ensemble kernel based on Thebelt et al., NeurIPS 2022.

Computes similarity as the mean leaf co-occurrence across trees in an ExtraTrees forest::

    k_tree(x, x') = (1/T) * sum_{t=1}^T  1[L_t(x) = L_t(x')]

The forest is fitted on observed data and naturally handles mixed continuous/discrete inputs
and conditional structure through its splits -- no explicit hierarchy specification is needed.
"""
from __future__ import annotations

from typing import Optional

import gpflow
import numpy as np
import tensorflow as tf
from sklearn.ensemble import ExtraTreesRegressor

from ....types import TensorType


class TreeEnsembleKernel(gpflow.kernels.Kernel):
    """A GPflow kernel that computes similarity via leaf co-occurrence in an ExtraTrees forest.

    This kernel has **no trainable TensorFlow parameters**. The forest is fitted externally
    via :meth:`fit_forest`, and the kernel matrix is computed from the leaf assignments.
    Wrap with ``gpflow.kernels.Constant() * TreeEnsembleKernel()`` to add a trainable
    output scale.

    :param n_estimators: Number of trees in the ExtraTrees forest.
    :param min_samples_leaf: Minimum number of samples at a leaf node.
    :param random_state: Random seed for the forest.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_leaf: int = 1,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        self._n_estimators = n_estimators
        self._min_samples_leaf = min_samples_leaf
        self._random_state = random_state
        self._forest: Optional[ExtraTreesRegressor] = None

    def fit_forest(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit (or refit) the ExtraTrees forest on observed data.

        :param X: Training inputs, shape ``[N, D]``.
        :param y: Training targets, shape ``[N]`` or ``[N, 1]``.
        """
        self._forest = ExtraTreesRegressor(
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            random_state=self._random_state,
        )
        self._forest.fit(X, y.ravel())

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> TensorType:
        """Compute the kernel matrix from leaf co-occurrence.

        :param X: First set of points, shape ``[N1, D]``.
        :param X2: Second set of points, shape ``[N2, D]``. If ``None``, uses ``X``.
        :return: Kernel matrix of shape ``[N1, N2]``.
        :raises RuntimeError: If the forest has not been fitted yet.
        """
        if self._forest is None:
            raise RuntimeError(
                "TreeEnsembleKernel: forest has not been fitted. "
                "Call fit_forest(X, y) before computing the kernel matrix."
            )

        X_np = np.asarray(X)
        leaves1 = self._forest.apply(X_np)  # [N1, T]

        if X2 is None:
            leaves2 = leaves1
        else:
            X2_np = np.asarray(X2)
            leaves2 = self._forest.apply(X2_np)  # [N2, T]

        # Co-occurrence: fraction of trees where both points land in the same leaf
        K_np = (leaves1[:, None, :] == leaves2[None, :, :]).mean(axis=-1)  # [N1, N2]

        return tf.constant(K_np, dtype=X.dtype if hasattr(X, "dtype") else tf.float64)

    def K_diag(self, X: TensorType) -> TensorType:
        """Self-similarity is always 1.0 (a point shares all leaves with itself).

        :param X: Points, shape ``[N, D]``.
        :return: Diagonal of shape ``[N]``, all ones.
        """
        return tf.ones([tf.shape(X)[0]], dtype=X.dtype if hasattr(X, "dtype") else tf.float64)
