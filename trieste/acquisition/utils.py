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
import copy
import functools
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import tensorflow as tf
from check_shapes import check_shapes

from ..data import Dataset
from ..models import ProbabilisticModelType
from ..observer import OBJECTIVE
from ..space import SearchSpaceType
from ..types import Tag, TensorType
from ..utils.misc import LocalizedTag
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

        # Use int64 to calculate the input tensor size, otherwise we can overflow for large tensors.
        elements_per_block = tf.size(x, out_type=tf.int64) / length
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


def copy_to_local_models(
    global_model: ProbabilisticModelType,
    num_local_models: int,
    key: Tag = OBJECTIVE,
) -> Mapping[Tag, ProbabilisticModelType]:
    """
    Helper method to copy a global model to local models.

    :param global_model: The global model.
    :param num_local_models: The number of local models to create.
    :param key: The tag prefix for the local models.
    :return: A mapping of the local models.
    """
    return {LocalizedTag(key, i): copy.deepcopy(global_model) for i in range(num_local_models)}


def with_local_datasets(
    datasets: Mapping[Tag, Dataset],
    num_local_datasets: int,
    local_dataset_indices: Optional[Sequence[TensorType]] = None,
) -> Dict[Tag, Dataset]:
    """
    Helper method to add local datasets if they do not already exist, by copying global datasets
    or a subset thereof.

    :param datasets: The original datasets.
    :param num_local_datasets: The number of local datasets to add per global tag.
    :param local_dataset_indices: Optional sequence of indices, indicating which parts of
        the global datasets should be copied. If None then the entire datasets are copied.
    :return: The updated mapping of datasets.
    """
    if local_dataset_indices is not None and len(local_dataset_indices) != num_local_datasets:
        raise ValueError(
            f"local_dataset_indices should have {num_local_datasets} entries, "
            f"has {len(local_dataset_indices)}"
        )

    updated_datasets = {}
    for tag in datasets:
        updated_datasets[tag] = datasets[tag]
        ltag = LocalizedTag.from_tag(tag)
        if not ltag.is_local:
            for i in range(num_local_datasets):
                target_ltag = LocalizedTag(ltag.global_tag, i)
                if target_ltag not in datasets:
                    if local_dataset_indices is None:
                        updated_datasets[target_ltag] = datasets[tag]
                    else:
                        # TODO: use sparse tensors instead
                        updated_datasets[target_ltag] = Dataset(
                            query_points=tf.gather(
                                datasets[tag].query_points, local_dataset_indices[i]
                            ),
                            observations=tf.gather(
                                datasets[tag].observations, local_dataset_indices[i]
                            ),
                        )

    return updated_datasets


@check_shapes(
    "points: [n_points, ...]",
    "return: [n_points]",
)
def get_unique_points_mask(points: TensorType, tolerance: float = 1e-6) -> TensorType:
    """Find the boolean mask of unique points in a tensor, within a given tolerance.

    Users can get the actual points with:

        mask = get_unique_points_mask(points, tolerance)
        unique_points = tf.boolean_mask(points, mask)

    Note: this uses a greedy parallel set covering algorithm, so isn't guaranteed to find the
    smallest possible set of unique points: e.g. for the points [[1],[2],[3]] with tolerance 1,
    it returns [True, False True] rather than [False, True, False].

    :param points: A tensor of points, with the first dimension being the number of points.
    :param tolerance: The tolerance up to which points are considered equal.
    :return: A boolean mask for the unique points.
    """

    # Calculate the pairwise "equality" between points
    pairwise_distances = tf.math.reduce_euclidean_norm(
        tf.expand_dims(points, 1) - tf.expand_dims(points, 0), axis=-1
    )
    pairwise_equal = pairwise_distances <= tolerance

    # Replace the upper triangle with False. That way, any row that is all False
    # is not equal to any point before it, so can be safely included in the output.
    upper_triangle = tf.linalg.band_part(tf.ones_like(pairwise_equal), 0, -1)
    triangle_equal = tf.logical_and(pairwise_equal, ~upper_triangle)

    # The converse is not true, however: a row that is not all False might still need
    # to be included if the only Trues in it correspond to other points that weren't selected.
    # For example, [1,2,3,4] with tolerance 1 should select not just 1 but also 3, because
    # even though 3 is equal to 2, it isn't equal to 1. We therefore keep track of which points
    # remain candidates for uniqueness, and repeat the triangle-based calculation with
    # those until there are no candidates left.
    candidate_mask = tf.ones(shape=tf.shape(points)[:1], dtype=tf.bool)
    unique_mask = tf.zeros(shape=tf.shape(points)[:1], dtype=tf.bool)
    while tf.reduce_any(candidate_mask):
        new_points = ~tf.reduce_any(triangle_equal, axis=1)
        unique_mask = tf.logical_or(unique_mask, new_points)
        equal_to_new_points = tf.reduce_any(pairwise_equal[new_points], axis=0)
        candidate_mask = tf.logical_and(candidate_mask, ~equal_to_new_points)
        # update triangle_equal to ignore non-candidate points
        triangle_equal = tf.where(candidate_mask, triangle_equal, tf.zeros_like(triangle_equal))
        triangle_equal = tf.where(
            tf.expand_dims(candidate_mask, 1), triangle_equal, tf.ones_like(triangle_equal)
        )

    return unique_mask
