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

import io
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, TypeVar, Union

import absl
import tensorflow as tf
from tensorflow.python.eager import context

from trieste.types import TensorType

if TYPE_CHECKING:
    import matplotlib


SummaryFilter = Callable[[str], bool]


def default_summary_filter(name: str) -> bool:
    """Default summary filter: omits any names that start with _."""
    return not (name.startswith("_") or "/_" in name)


_TENSORBOARD_WRITER: Optional[tf.summary.SummaryWriter] = None
_STEP_NUMBER: int = 0
_SUMMARY_FILTER: SummaryFilter = default_summary_filter


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
def tensorboard_writer(summary_writer: Optional[tf.summary.SummaryWriter]) -> Iterator[None]:
    """
    A context manager for setting or overriding a TensorBoard summary writer inside a code block.

    :param summary_writer: optional summary writer instance.
    """
    old_writer = get_tensorboard_writer()
    set_tensorboard_writer(summary_writer)
    yield
    set_tensorboard_writer(old_writer)


def set_step_number(new_step_number: int) -> None:
    """
    Set an optimization step number to use for logging purposes.

    :param new_step_number: new current step number
    :raise ValueError: if step_number < 0
    """
    global _STEP_NUMBER
    _STEP_NUMBER = new_step_number


def get_step_number() -> int:
    """
    Get the optimization step number used for logging purposes.

    :return: current step number.
    """
    return _STEP_NUMBER


@contextmanager
def step_number(new_step_number: int) -> Iterator[None]:
    """
    A context manager for setting or overriding the optimization step number inside a code block.

    :param new_step_number: new current step number
    """
    old_step_number = get_step_number()
    set_step_number(new_step_number)
    yield
    set_step_number(old_step_number)


def set_summary_filter(summary_filter: SummaryFilter) -> None:
    """
    Set a filter on summary names. The default is to only omit names that start with _.

    :param summary_filter: new summary filter
    """
    global _SUMMARY_FILTER
    _SUMMARY_FILTER = summary_filter


def get_summary_filter() -> SummaryFilter:
    """
    Get the current filter on summary names. The default is to only omit names that start with _.

    :return: current summary filter.
    """
    return _SUMMARY_FILTER


def get_current_name_scope() -> str:
    """Returns the full name scope. Copied from TF 2.5."""
    ctx = context.context()
    if ctx.executing_eagerly():
        return ctx.scope_name.rstrip("/")
    else:
        return tf.compat.v1.get_default_graph().get_name_scope()


def include_summary(name: str) -> bool:
    """
    Whether a summary name should be included.

    :param: full summary name (including name spaces)
    :return: whether the summary should be included.
    """
    full_name = get_current_name_scope() + "/" + name
    return _SUMMARY_FILTER(full_name)


T = TypeVar("T")


def evaluate_data(data: T | Callable[[], T]) -> T:
    """Return the passed in data, evaluating it if it's inside a closure."""
    return data() if callable(data) else data


def histogram(name: str, data: TensorType | Callable[[], TensorType], **kwargs: Any) -> bool:
    """
    Wrapper for tf.summary.histogram that first filters out unwanted summaries by name.
    Accepts either data or closures that only get evaluated when logged.
    """
    if include_summary(name):
        try:
            return tf.summary.histogram(name, evaluate_data(data), **kwargs)
        except Exception as e:
            tf.print(
                f"Failed to write tensorboard histogram summary '{name}':\n\n{e}",
                output_stream=absl.logging.INFO,
            )
    return False


def scalar(name: str, data: float | Callable[[], float], **kwargs: Any) -> bool:
    """
    Wrapper for tf.summary.scalar that first filters out unwanted summaries by name.
    Accepts either data or closures that only get evaluated when logged.
    """
    if include_summary(name):
        try:
            return tf.summary.scalar(name, evaluate_data(data), **kwargs)
        except Exception as e:
            tf.print(
                f"Failed to write tensorboard scalar summary '{name}':\n\n{e}",
                output_stream=absl.logging.INFO,
            )
    return False


def text(name: str, data: str | Callable[[], str], **kwargs: Any) -> bool:
    """
    Wrapper for tf.summary.text that first filters out unwanted summaries by name.
    Accepts either data or closures that only get evaluated when logged.
    """
    if include_summary(name):
        try:
            return tf.summary.text(name, evaluate_data(data), **kwargs)
        except Exception as e:
            tf.print(
                f"Failed to write tensorboard text summary '{name}':\n\n{e}",
                output_stream=absl.logging.INFO,
            )
    return False


def pyplot(
    name: str, figure: Union["matplotlib.figure.Figure", Callable[[], "matplotlib.figure.Figure"]]
) -> bool:
    """
    Utility function for passing a matplotlib figure to tf.summary.image.
    Accepts either data or closures that only get evaluated when logged.
    """
    if include_summary(name):
        try:
            figure = evaluate_data(figure)
            with io.BytesIO() as buffer:
                figure.savefig(buffer, dpi=150.0, format="png")
                buffer.seek(0)
                image = tf.image.decode_png(buffer.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            return tf.summary.image(name, image)
        except Exception as e:
            tf.print(
                f"Failed to write tensorboard image summary '{name}':\n\n{e}",
                output_stream=absl.logging.INFO,
            )
    return False
