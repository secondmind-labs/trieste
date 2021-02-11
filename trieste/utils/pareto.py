# Copyright 2020 The Trieste Contributors
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

import tensorflow as tf

from trieste.type import TensorType


def non_dominated_sort(observations: TensorType) -> TensorType:
    """
    Computes the non-dominated set for a set of data points

    :param dataset: A :class:`~trieste.data.Dataset` of observed points
    :return: tuple of the non-dominated set and the degree of dominance,
        dominances gives the number of dominating points for each data point
    """
    extended = tf.tile(observations[None], [len(observations), 1, 1])
    swapped_ext = tf.transpose(extended, [1, 0, 2])
    dominance = tf.math.count_nonzero(
        tf.logical_and(
            tf.reduce_all(extended <= swapped_ext, axis=2),
            tf.reduce_any(extended < swapped_ext, axis=2),
        ),
        axis=1,
    )

    return tf.boolean_mask(observations, dominance == 0), dominance
