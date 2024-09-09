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

from check_shapes import check_shapes

from ..data import Dataset
from ..observer import OBJECTIVE, MultiObserver, Observer, SingleObserver
from ..types import Tag, TensorType
from ..utils.misc import LocalizedTag, flatten_leading_dims


@overload
def mk_observer(objective: Callable[[TensorType], TensorType]) -> SingleObserver: ...


@overload
def mk_observer(objective: Callable[[TensorType], TensorType], key: Tag) -> MultiObserver: ...


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
    objective_or_observer: Union[Callable[[TensorType], TensorType], Observer],
    default_key: Tag = OBJECTIVE,
) -> MultiObserver:
    """
    Create an observer that returns the data from ``objective`` or an existing ``observer``
    separately for each query point in a batch.

    :param objective_or_observer: An objective or an existing observer.
    :param default_key: The default key to use if ``objective_or_observer`` is an objective or
        does not return a mapping.
    :return: A multi-observer across the batch dimension of query points, returning the data from
        ``objective_or_observer``.
    """

    @check_shapes("qps: [n_points, batch_size, n_dims]")
    def _observer(qps: TensorType) -> Mapping[Tag, Dataset]:
        # Call objective with rank 2 query points by flattening batch dimension.
        # Some objectives might only expect rank 2 query points, so this is safer.
        batch_size = qps.shape[1]
        flat_qps, unflatten = flatten_leading_dims(qps)
        obs_or_dataset = objective_or_observer(flat_qps)

        if not isinstance(obs_or_dataset, (Mapping, Dataset)):
            # Just a single observation, so wrap in a dataset.
            obs_or_dataset = Dataset(flat_qps, obs_or_dataset)

        if isinstance(obs_or_dataset, Dataset):
            # Convert to a mapping with a default key.
            obs_or_dataset = {default_key: obs_or_dataset}

        datasets = {}
        for key, dataset in obs_or_dataset.items():
            # Include overall dataset and per batch dataset.
            flat_obs = dataset.observations
            qps = unflatten(flat_qps)
            obs = unflatten(flat_obs)
            datasets[key] = dataset
            for i in range(batch_size):
                datasets[LocalizedTag(key, i)] = Dataset(qps[:, i], obs[:, i])

        return datasets

    return _observer
