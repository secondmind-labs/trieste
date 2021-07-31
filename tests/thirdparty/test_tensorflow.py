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
import pytest
import tensorflow as tf


def test_raise_is_incompatible_with_tf_function() -> None:
    err_msg = "very specific error message 13579"

    @tf.function
    def f(a: tf.Tensor) -> tf.Tensor:
        if a <= tf.constant(0):
            raise ValueError(err_msg)

        return a

    with pytest.raises(ValueError, match=err_msg):
        f(tf.constant(1))  # note that 1 should *not* trigger the error branch, but does


def test_tf_debugging_is_compatible_with_tf_function() -> None:
    err_msg = "very specific error message 2468"

    @tf.function
    def f(a: tf.Tensor) -> tf.Tensor:
        tf.debugging.assert_positive(a, message=err_msg)
        return a

    f(tf.constant(1))

    with pytest.raises(tf.errors.InvalidArgumentError, match=err_msg):
        f(tf.constant(-1))
