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

"""
This module contains functions convenient for creating :class:`Observer` objects that return data
from objective functions, appropriately formatted for usage with the toolbox.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Mapping, Optional, Union, overload

import tensorflow as tf
from check_shapes import check_shapes

from ..data import Dataset
from ..observer import MultiObserver, Observer, SingleObserver
from ..types import Tag, TensorType
from ..utils.misc import LocalTag


@overload
def mk_observer(objective: Callable[[TensorType], TensorType]) -> SingleObserver:
    ...


@overload
def mk_observer(objective: Callable[[TensorType], TensorType], key: Tag) -> MultiObserver:
    ...


def mk_observer(
    objective: Callable[[TensorType], TensorType], key: Optional[Tag] = None
) -> Observer:
    """
    :param objective: An objective function designed to be used with a single data set and model.
    :param key: An optional key to use to access the data from the observer result.
    :return: An observer returning the data from ``objective``.
    """
    if key is not None:
        return lambda qp: {key: Dataset(qp, objective(qp))}
    else:
        return lambda qp: Dataset(qp, objective(qp))


def mk_multi_observer(**kwargs: Callable[[TensorType], TensorType]) -> MultiObserver:
    """
    :param kwargs: Observation functions.
    :return: An multi-observer returning the data from ``kwargs``.
    """
    return lambda qp: {key: Dataset(qp, objective(qp)) for key, objective in kwargs.items()}


def mk_batch_observer(
    objective_or_observer: Union[Callable[[TensorType], TensorType], SingleObserver],
    key: Optional[Tag] = None,
) -> Observer:
    """
    Create an observer that returns the data from ``objective`` or an existing ``observer``
    separately for each query point in a batch.

    :param objective_or_observer: An objective or an existing observer designed to be used with a
        single data set and model.
    :param key: An optional key to use to access the data from the observer result.
    :return: A multi-observer across the batch dimension of query points, returning the data from
        ``objective``. If ``key`` is provided, the observer will be a mapping. Otherwise, it will
        return a single dataset.
    :raise ValueError (or tf.errors.InvalidArgumentError): If ``objective_or_observer`` is a
        multi-observer.
    """

    @check_shapes("qps: [n_points, batch_size, n_dims]")
    # Note that the return type is not correct, but that is what mypy is happy with. It should be
    # Mapping[Tag, Dataset] if key is not None, otherwise Dataset.
    # One solution is to create two separate functions, but that will result in some duplicate code.
    def _observer(qps: TensorType) -> Mapping[Tag, Dataset]:
        # Call objective with rank 2 query points by flattening batch dimension.
        # Some objectives might only expect rank 2 query points, so this is safer.
        batch_size = qps.shape[1]
        qps = tf.reshape(qps, [-1, qps.shape[-1]])
        obs_or_dataset = objective_or_observer(qps)

        if isinstance(obs_or_dataset, Mapping):
            raise ValueError("mk_batch_observer does not support multi-observers")
        elif not isinstance(obs_or_dataset, Dataset):
            obs_or_dataset = Dataset(qps, obs_or_dataset)

        if key is None:
            # Always use rank 2 shape as models (e.g. GPR) expect this, so return as is.
            return obs_or_dataset
        else:
            # Include overall dataset and per batch dataset.
            obs = obs_or_dataset.observations
            qps = tf.reshape(qps, [-1, batch_size, qps.shape[-1]])
            obs = tf.reshape(obs, [-1, batch_size, obs.shape[-1]])
            datasets: Mapping[Tag, Dataset] = {
                key: obs_or_dataset,
                **{LocalTag(key, i): Dataset(qps[:, i], obs[:, i]) for i in range(batch_size)},
            }
            return datasets

    return _observer
