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
from __future__ import annotations

import tempfile

import pytest
import tensorflow as tf

from trieste.logging import (
    get_step_number,
    get_tensorboard_writer,
    set_step_number,
    set_tensorboard_writer,
    step_number,
    tensorboard_writer,
)


def test_get_tensorboard_writer_default() -> None:
    assert get_tensorboard_writer() is None


def test_set_get_tensorboard_writer() -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        summary_writer = tf.summary.create_file_writer(tmpdirname)
        set_tensorboard_writer(summary_writer)
        assert get_tensorboard_writer() is summary_writer
        set_tensorboard_writer(None)
        assert get_tensorboard_writer() is None


def test_tensorboard_writer() -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        summary_writer = tf.summary.create_file_writer(tmpdirname)
        assert get_tensorboard_writer() is None
        with tensorboard_writer(summary_writer):
            assert get_tensorboard_writer() is summary_writer
            with tensorboard_writer(None):
                assert get_tensorboard_writer() is None
            assert get_tensorboard_writer() is summary_writer
        assert get_tensorboard_writer() is None


@pytest.mark.parametrize("step", [0, 1, 42])
def test_set_get_step_number(step: int) -> None:
    set_step_number(step)
    assert get_step_number() == step
    set_step_number(0)
    assert get_step_number() == 0


def test_set_step_number_error() -> None:
    with pytest.raises(ValueError):
        set_step_number(-1)


@pytest.mark.parametrize("step", [0, 1, 42])
def test_step_number(step: int) -> None:
    assert get_step_number() == 0
    with step_number(step):
        assert get_step_number() == step
        with step_number(0):
            assert get_step_number() == 0
        assert get_step_number() == step
    assert get_step_number() == 0
