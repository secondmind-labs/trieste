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
    :returns Split acquisition function.
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
