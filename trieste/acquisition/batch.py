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

import tensorflow as tf

from ..types import TensorType
from .interface import AcquisitionFunction


def batchify_acquisition_function(
    fn: AcquisitionFunction,
    batch_size: int,
) -> AcquisitionFunction:
    """
    A wrapper around an :const:`AcquisitionFunction` to split its input into batches.
    Splits `x` into batches along the first dimension, calls `fn` on each batch, and then stitches
    the results back together, so that it looks like `fn` was called with all of `x` in one batch.
    :param fn: Acquisition function to call with batches of data.
    :param batch_size: Call fn with tensors of at most this size.
    :returns Batched acquisition function.
    """
    assert batch_size > 0, f"Batch size has to be positive integer! Found {batch_size}."

    @functools.wraps(fn)
    def wrapper(x: TensorType) -> TensorType:
        x = tf.convert_to_tensor(x)

        # this currently assumes leading dimension of x is batch dimension.
        length = x.shape[0]
        if length == 0:
            return fn(x)

        elements_per_block = tf.size(x) / length
        blocks_per_batch = tf.cast(tf.math.ceil(batch_size / elements_per_block), tf.int32)

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
