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

r"""
This file contains wrappers for some implementations of basic GPflux architectures.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math
import torch as t
import torch.nn as nn
import bayesfunc as bf

import numpy as np
import tensorflow as tf

from ...space import Box
from ...types import TensorType


def build_sqexp_deep_inv_wishart(
    query_points: TensorType,
    num_layers: int,
    num_inducing: int,
    observations: Optional[TensorType] = None,
    likelihood_noise_variance: float = 1e-2,
    search_space: Optional[Box] = None,
) -> nn.Module:
    tf.debugging.assert_rank(
        query_points, 2, message="For this architecture, the rank of the input data must be 2."
    )
    num_data, input_dim = query_points.shape

    # Input data to model must be np.ndarray for k-means algorithm
    if isinstance(query_points, tf.Tensor):
        query_points = query_points.numpy()

    # Pad query_points with additional random values to provide enough inducing points
    if num_inducing > len(query_points):
        if search_space is not None:
            if not isinstance(search_space, Box):
                raise ValueError("Currently only `Box` instances are supported for `search_space`.")
            additional_points = search_space.sample_sobol(num_inducing - len(query_points)).numpy()
        else:
            additional_points = np.random.randn(
                num_inducing - len(query_points), *query_points.shape[1:]
            )
        query_points = np.concatenate([query_points, additional_points], 0)

    if observations is not None:
        if tf.shape(observations)[-1] != 1:
            raise ValueError("Output dim must be 1 for this model")
        if tf.shape(observations)[0] != num_data:
            raise ValueError("Received a different number of query points and observations")
        if isinstance(observations, tf.Tensor):
            observations = observations.numpy()

        if num_inducing > len(observations):
            observations = np.concatenate(
                [
                    observations,
                    np.random.randn(num_inducing - len(observations), *observations.shape[1:]),
                ],
                0,
            )

    inducing_init = t.from_numpy(query_points[:num_inducing])
    inducing_targets = t.from_numpy(observations[:num_inducing]) if observations is not None else None

    def layer(num_inducing: int) -> nn.Sequential:
        return nn.Sequential(
            bf.SqExpKernelGram(trainable_noise=True), bf.IWLayer(inducing_batch=num_inducing)
        )

    class ConstantLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.constant = nn.Parameter(t.zeros(()))

        def forward(self, x):
            return x + self.constant

    net = nn.Sequential(
        bf.FeaturesToKernelARD(inducing_batch=num_inducing, in_features=input_dim),
        bf.SqExpKernelGram(trainable_noise=True, lengthscale=False),
        bf.IWLayer(inducing_batch=num_inducing),
        *[layer(num_inducing) for _ in range(num_layers - 2)],
        bf.SqExpKernelGram(),
        bf.GIGP(out_features=1, inducing_targets=inducing_targets, inducing_batch=num_inducing)
    )

    net = bf.InducingWrapper(net, inducing_batch=num_inducing, inducing_data=inducing_init)
    return nn.Sequential(net, ConstantLayer(), bf.NormalLearnedScale(log_scale=math.sqrt(likelihood_noise_variance)))
