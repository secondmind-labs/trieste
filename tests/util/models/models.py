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
Utility functions for tests.
"""

from __future__ import annotations

import tensorflow as tf


def fnc_3x_plus_10(x: tf.Tensor) -> tf.Tensor:
    return 3.0 * x + 10


def fnc_2sin_x_over_3(x: tf.Tensor) -> tf.Tensor:
    return 2.0 * tf.math.sin(x / 3.0)


def binary_line(x: tf.Tensor) -> tf.Tensor:
    return tf.stack([1 if xi > 0 else 0 for xi in x])
