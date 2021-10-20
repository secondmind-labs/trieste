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
""" This module contains logging utilities. """
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import tensorflow as tf

_TENSORBOARD_WRITER: Optional[tf.summary.SummaryWriter] = None


def set_tensorboard_writer(summary_writer: Optional[tf.summary.SummaryWriter]) -> None:
    """
    Set a :class:`~tf.summary.SummaryWriter` instance to use for logging
    to TensorBoard, or `None` to disable.

    :param summary_writer: optional summary writer instance.
    """
    global _TENSORBOARD_WRITER
    _TENSORBOARD_WRITER = summary_writer


def get_tensorboard_writer() -> Optional[tf.summary.SummaryWriter]:
    """
    Returns a :class:`~tf.summary.SummaryWriter` instance to use for logging
    to TensorBoard, or `None`.

    :return: optional summary writer instance.
    """
    return _TENSORBOARD_WRITER


@contextmanager
def using_tensorboard_writer(summary_writer: Optional[tf.summary.SummaryWriter]) -> Iterator[None]:
    """
    A context manager for setting or overriding a TensorBoard summary writer inside a code block.

    :param summary_writer: optional summary writer instance.
    """
    old_writer = get_tensorboard_writer()
    set_tensorboard_writer(summary_writer)
    yield
    set_tensorboard_writer(old_writer)
