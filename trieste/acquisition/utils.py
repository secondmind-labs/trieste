# Copyright 2022 The Trieste Contributors
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
import functools
from typing import Tuple, Union

import tensorflow as tf
from check_shapes import check_shapes

from ..data import Dataset
from ..space import SearchSpaceType
from ..types import TensorType
from .interface import AcquisitionFunction
from .optimizer import AcquisitionOptimizer


def split_acquisition_function(
    fn: AcquisitionFunction,
    split_size: int,
) -> AcquisitionFunction:
    """
    A wrapper around an :const:`AcquisitionFunction` to split its input into batches.
    Splits `x` into batches along the first dimension, calls `fn` on each batch, and then stitches
    the results back together, so that it looks like `fn` was called with all of `x` in one batch.
    :param fn: Acquisition function to split.
    :param split_size: Call fn with tensors of at most this size.
    :return: Split acquisition function.
    """
    if split_size <= 0:
        raise ValueError(f"split_size must be positive, got {split_size}")

    @functools.wraps(fn)
    def wrapper(x: TensorType) -> TensorType:
        x = tf.convert_to_tensor(x)

        # this currently assumes leading dimension of x is the split dimension.
        length = x.shape[0]
        if length == 0:
            return fn(x)

        elements_per_block = tf.size(x) / length
        blocks_per_batch = tf.cast(tf.math.ceil(split_size / elements_per_block), tf.int32)

        num_batches = tf.cast(tf.math.ceil(length / blocks_per_batch) - 1, tf.int32)
        batch_sizes = tf.concat(
            [
                tf.ones(num_batches, tf.int32) * blocks_per_batch,
                [length - num_batches * blocks_per_batch],
            ],
            axis=0,
        )

        if batch_sizes.shape[0] <= 1:
            return fn(x)

        batch_inputs = tf.split(x, batch_sizes)

        batch_outputs = []
        for batch_input in batch_inputs:
            output = fn(batch_input)
            batch_outputs.append(output)

        return tf.concat(batch_outputs, axis=0)

    return wrapper


def split_acquisition_function_calls(
    optimizer: AcquisitionOptimizer[SearchSpaceType],
    split_size: int,
) -> AcquisitionOptimizer[SearchSpaceType]:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` so that evaluations of the acquisition functions
    are split into batches on the first dimension and then stitched back together.
    This can be useful to reduce memory usage when evaluating functions over large spaces.

    :param optimizer: An optimizer that returns batches of points with shape [V, ...].
    :param split_size: The desired maximum number of points in acquisition function evaluations.
    :return: An :const:`AcquisitionOptimizer` that still returns points with the shape [V, ...]
        but evaluates at most split_size points at a time.
    """
    if split_size <= 0:
        raise ValueError(f"split_size must be positive, got {split_size}")

    def split_optimizer(
        search_space: SearchSpaceType,
        f: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    ) -> TensorType:
        af, n = f if isinstance(f, tuple) else (f, 1)
        taf = split_acquisition_function(af, split_size)
        return optimizer(search_space, (taf, n) if isinstance(f, tuple) else taf)

    return split_optimizer


def select_nth_output(x: TensorType, output_dim: int = 0) -> TensorType:
    """
    A utility function for trajectory sampler-related acquisition functions which selects the `n`th
    output as the trajectory to be used, with `n` specified by ``output_dim``. Defaults to the first
    output.

    :param x: Input with shape [..., B, L], where L is the number of outputs of the model.
    :param output_dim: Dimension of the output to be selected. Defaults to the first output.
    :return: TensorType with shape [..., B], where the output_dim dimension has been selected to
        reduce the input.
    """
    return x[..., output_dim]


def get_local_dataset(local_space: SearchSpaceType, dataset: Dataset) -> Dataset:
    """
    A utility function that takes in a dataset and returns the entries lying
    within a given search space.

    :param local_space: A search space.
    :param dataset: A Dataset.
    :return: A Dataset containing entries only in the local_space.
    """
    if tf.shape(dataset.query_points)[1] != local_space.dimension:
        raise ValueError("Dataset and search space must have equal dimensions")

    is_in_region_mask = local_space.contains(dataset.query_points)
    local_dataset = Dataset(
        query_points=tf.boolean_mask(dataset.query_points, is_in_region_mask),
        observations=tf.boolean_mask(dataset.observations, is_in_region_mask),
    )
    return local_dataset


@check_shapes(
    "points: [n_points, ...]",
    "return: [n_points]",
)
def get_unique_points_mask(points: TensorType, tolerance: float = 1e-6) -> TensorType:
    """Find the boolean mask of unique points in a tensor, within a given tolerance.

    Users can get the actual points with:

        mask = get_unique_points_mask(points, tolerance)
        unique_points = tf.boolean_mask(points, mask)

    :param points: A tensor of points, with the first dimension being the number of points.
    :param tolerance: The tolerance within which points are considered equal.
    :return: A boolean mask for the unique points.
    """

    tolerance = tf.constant(tolerance, dtype=points.dtype)
    n_points = tf.shape(points)[0]
    mask = tf.zeros(shape=(n_points,), dtype=tf.bool)

    for idx in tf.range(n_points):
        # Pairwise distance with previous unique points.
        used_points = tf.boolean_mask(points, mask)
        distances = tf.norm(points[idx] - used_points, axis=-1)
        # Find if there is any point within the tolerance.
        min_distance = tf.reduce_min(distances)

        # Update mask.
        is_unique_point = min_distance >= tolerance
        mask = tf.tensor_scatter_nd_update(mask, [[idx]], [is_unique_point])

    return mask
