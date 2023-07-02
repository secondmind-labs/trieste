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
from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from types import TracebackType
from typing import Any, Callable, Generic, Mapping, NoReturn, Optional, Tuple, Type, TypeVar

import numpy as np
import tensorflow as tf
from tensorflow import DType
from tensorflow.python.util import nest
from typing_extensions import Final, final

from ..types import TensorType

C = TypeVar("C", bound=Callable[..., object])
""" A type variable bound to `typing.Callable`. """


def jit(apply: bool = True, **optimize_kwargs: Any) -> Callable[[C], C]:
    """
    A decorator that conditionally wraps a function with `tf.function`.

    :param apply: If `True`, the decorator is equivalent to `tf.function`. If `False`, the decorator
        does nothing.
    :param optimize_kwargs: Additional arguments to `tf.function`.
    :return: The decorator.
    """

    def decorator(func: C) -> C:
        return tf.function(func, **optimize_kwargs) if apply else func

    return decorator


def shapes_equal(this: TensorType, that: TensorType) -> TensorType:
    """
    Return a scalar tensor containing: `True` if ``this`` and ``that`` have equal runtime shapes,
    else `False`.
    """
    return tf.rank(this) == tf.rank(that) and tf.reduce_all(tf.shape(this) == tf.shape(that))


def to_numpy(t: TensorType) -> "np.ndarray[Any, Any]":
    """
    :param t: An array-like object.
    :return: ``t`` as a NumPy array.
    """
    if isinstance(t, tf.Tensor):
        return t.numpy()

    return t


ResultType = TypeVar("ResultType", covariant=True)
""" An unbounded covariant type variable. """


class Result(Generic[ResultType], ABC):
    """
    Represents the result of an operation that can fail with an exception. It contains either the
    operation return value (in an :class:`Ok`), or the exception raised (in an :class:`Err`).

    To check whether instances such as

        >>> res = Ok(1)
        >>> other_res = Err(ValueError("whoops"))

    contain a value, use :attr:`is_ok` (or :attr:`is_err`)

        >>> res.is_ok
        True
        >>> other_res.is_ok
        False

    We can access the value if it :attr:`is_ok` using :meth:`unwrap`.

        >>> res.unwrap()
        1

    Trying to access the value of a failed :class:`Result`, or :class:`Err`, will raise the wrapped
    exception

        >>> other_res.unwrap()
        Traceback (most recent call last):
            ...
        ValueError: whoops

    **Note:** This class is not intended to be subclassed other than by :class:`Ok` and
    :class:`Err`.
    """

    @property
    @abstractmethod
    def is_ok(self) -> bool:
        """`True` if this :class:`Result` contains a value, else `False`."""

    @property
    def is_err(self) -> bool:
        """
        `True` if this :class:`Result` contains an error, else `False`. The opposite of
        :attr:`is_ok`.
        """
        return not self.is_ok

    @abstractmethod
    def unwrap(self) -> ResultType:
        """
        :return: The contained value, if it exists.
        :raise Exception: If there is no contained value.
        """


@final
class Ok(Result[ResultType]):
    """Wraps the result of a successful evaluation."""

    def __init__(self, value: ResultType):
        """
        :param value: The result of a successful evaluation.
        """
        self._value = value

    def __repr__(self) -> str:
        """"""
        return f"Ok({self._value!r})"

    @property
    def is_ok(self) -> bool:
        """`True` always."""
        return True

    def unwrap(self) -> ResultType:
        """
        :return: The wrapped value.
        """
        return self._value


@final
class Err(Result[NoReturn]):
    """Wraps the exception that occurred during a failed evaluation."""

    def __init__(self, exc: Exception):
        """
        :param exc: The exception that occurred.
        """
        self._exc = exc

    def __repr__(self) -> str:
        """"""
        return f"Err({self._exc!r})"

    @property
    def is_ok(self) -> bool:
        """`False` always."""
        return False

    def unwrap(self) -> NoReturn:
        """
        :raise Exception: Always. Raises the wrapped exception.
        """
        raise self._exc


class DEFAULTS:
    """Default constants used in Trieste."""

    JITTER: Final[float] = 1e-6
    """
    The default jitter, typically used to stabilise computations near singular points, such as in
    Cholesky decomposition.
    """


K = TypeVar("K")
""" An unbound type variable. """

U = TypeVar("U")
""" An unbound type variable. """

V = TypeVar("V")
""" An unbound type variable. """


def map_values(f: Callable[[U], V], mapping: Mapping[K, U]) -> Mapping[K, V]:
    """
    Apply ``f`` to each value in ``mapping`` and return the result. If ``f`` does not modify its
    argument, :func:`map_values` does not modify ``mapping``. For example:

    >>> import math
    >>> squares = {'a': 1, 'b': 4, 'c': 9}
    >>> map_values(math.sqrt, squares)['b']
    2.0
    >>> squares
    {'a': 1, 'b': 4, 'c': 9}

    :param f: The function to apply to the values in ``mapping``.
    :param mapping: A mapping.
    :return: A new mapping, whose keys are the same as ``mapping``, and values are the result of
        applying ``f`` to each value in ``mapping``.
    """
    return {k: f(u) for k, u in mapping.items()}


class Timer:
    """
    Functionality for timing chunks of code. For example:
    >>> from time import sleep
    >>> with Timer() as timer: sleep(2.0)
    >>> timer.time  # doctest: +SKIP
    2.0
    """

    def __enter__(self) -> Timer:
        self.start = perf_counter()
        return self

    def __exit__(
        self,
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.end = perf_counter()
        self.time = self.end - self.start


def flatten_leading_dims(
    x: TensorType, output_dims: int = 2
) -> Tuple[TensorType, Callable[[TensorType], TensorType]]:
    """
    Flattens the leading dimensions of `x` (all but the last `output_dims` dimensions), and returns
    a function that can be used to restore them (typically after first manipulating the
    flattened tensor).
    """
    tf.debugging.assert_positive(output_dims, message="output_dims must be positive")
    tf.debugging.assert_less_equal(
        output_dims, tf.rank(x), message="output_dims must no greater than tensor rank"
    )

    x_batched_shape = tf.shape(x)
    batch_shape = x_batched_shape[: -output_dims + 1] if output_dims > 1 else x_batched_shape
    input_shape = x_batched_shape[-output_dims + 1 :] if output_dims > 1 else []
    x_flat_shape = tf.concat([[-1], input_shape], axis=0)

    def unflatten(y: TensorType) -> TensorType:
        y_flat_shape = tf.shape(y)
        output_shape = y_flat_shape[1:]
        y_batched_shape = tf.concat([batch_shape, output_shape], axis=0)
        y_batched = tf.reshape(y, y_batched_shape)
        return y_batched

    return tf.reshape(x, x_flat_shape), unflatten


def get_variables(object: Any) -> tuple[tf.Variable, ...]:
    """
    Return the sequence of variables in an object.

    This is essentially a reimplementation of the `variables` property of tf.Module
    but doesn't require that we, or any of our substructures, inherit from that.

    :return: A sequence of variables of the object (sorted by attribute
      name) followed by variables from all submodules recursively (breadth
      first).
    """

    def _is_variable(obj: Any) -> bool:
        return isinstance(obj, tf.Variable)

    return tuple(_flatten(object, predicate=_is_variable, expand_composites=True))


_TF_MODULE_IGNORED_PROPERTIES = frozenset(
    ("_self_unconditional_checkpoint_dependencies", "_self_unconditional_dependency_names")
)


def _flatten(  # type: ignore[no-untyped-def]
    model,
    recursive=True,
    predicate=None,
    attribute_traversal_key=None,
    with_path=False,
    expand_composites=False,
):
    """
    Flattened attribute values in sorted order by attribute name.

    This is taken verbatim from tensorflow core but uses a modified _flatten_module.
    """
    if predicate is None:
        predicate = lambda _: True  # noqa: E731

    return _flatten_module(
        model,
        recursive=recursive,
        predicate=predicate,
        attributes_to_ignore=_TF_MODULE_IGNORED_PROPERTIES,
        attribute_traversal_key=attribute_traversal_key,
        with_path=with_path,
        expand_composites=expand_composites,
    )


def _flatten_module(  # type: ignore[no-untyped-def]
    module,
    recursive,
    predicate,
    attribute_traversal_key,
    attributes_to_ignore,
    with_path,
    expand_composites,
    module_path=(),
    seen=None,
):
    """
    Implementation of `flatten`.

    This is a reimplementation of the equivalent function in tf.Module so
    that we can extract the list of variables from a Trieste model wrapper
    without the need to inherit from it.
    """
    if seen is None:
        seen = {id(module)}

    # [CHANGED] Differently from the original version, here we catch an exception
    # as some of the components of the wrapper do not implement __dict__
    try:
        module_dict = vars(module)
    except TypeError:
        module_dict = {}

    submodules = []

    for key in sorted(module_dict, key=attribute_traversal_key):
        if key in attributes_to_ignore:
            continue

        prop = module_dict[key]
        try:
            leaves = nest.flatten_with_tuple_paths(prop, expand_composites=expand_composites)
        except Exception:  # pylint: disable=broad-except
            leaves = []

        for leaf_path, leaf in leaves:
            leaf_path = (key,) + leaf_path

            if not with_path:
                leaf_id = id(leaf)
                if leaf_id in seen:
                    continue
                seen.add(leaf_id)

            if predicate(leaf):
                if with_path:
                    yield module_path + leaf_path, leaf
                else:
                    yield leaf

            # [CHANGED] Differently from the original, here we skip checking whether the leaf
            # is a module, since the trieste models do NOT inherit from tf.Module
            if recursive:  # and _is_module(leaf):
                # Walk direct properties first then recurse.
                submodules.append((module_path + leaf_path, leaf))

    for submodule_path, submodule in submodules:
        subvalues = _flatten_module(
            submodule,
            recursive=recursive,
            predicate=predicate,
            attribute_traversal_key=attribute_traversal_key,
            attributes_to_ignore=_TF_MODULE_IGNORED_PROPERTIES,
            with_path=with_path,
            expand_composites=expand_composites,
            module_path=submodule_path,
            seen=seen,
        )

        for subvalue in subvalues:
            # Predicate is already tested for these values.
            yield subvalue


def tf_argmin_with_tie_breaks(
    input: TensorType, axis: int = 0, output_type: DType = tf.int64
) -> TensorType:
    """
    Returns the index with the smallest value across axes of a tensor.

    Identical to tf.math.argmin, except ties are resolved randomly.

    :param input: Input tensor.
    :param axis: Axis.
    :param output_type: Optional `tf.Dtype` from: `tf.int32, tf.int64`.
    :return: A tensor of type `output_type`.
    """
    return _tf_argminmax_with_tie_breaks(True, input, axis, output_type)


def tf_argmax_with_tie_breaks(
    input: TensorType, axis: int = 0, output_type: DType = tf.int64
) -> TensorType:
    """
    Returns the index with the largest value across axes of a tensor.

    Identical to tf.math.argmax, except ties are resolved randomly.

    :param input: Input tensor.
    :param axis: Axis.
    :param output_type: Optional `tf.Dtype` from: `tf.int32, tf.int64`.
    :return: A tensor of type `output_type`.
    """
    return _tf_argminmax_with_tie_breaks(False, input, axis, output_type)


def _tf_argminmax_with_tie_breaks(
    argmin: bool, input: TensorType, axis: int, output_type: DType
) -> TensorType:
    # transpose axis to end and flatten
    shape = tf.shape(input)
    range = tf.range(tf.rank(input), dtype=output_type)
    transpose_perm = tf.concat([range[:axis], range[axis + 1 :], range[axis : axis + 1]], 0)
    transposed = tf.transpose(input, transpose_perm)
    flattened = tf.reshape(transposed, [-1, shape[axis]])

    # shuffle along last axis and call argmin/argmax
    shuffle_perm = tf.map_fn(
        lambda x: tf.random.shuffle(tf.range(shape[axis], dtype=output_type)),
        flattened,
        fn_output_signature=output_type,
    )
    shuffled = tf.map_fn(
        lambda x: tf.gather(x[0], x[1]), (flattened, shuffle_perm), fn_output_signature=input.dtype
    )
    tf_argminmax = tf.math.argmin if argmin else tf.math.argmax
    argminmax = tf_argminmax(shuffled, axis=-1, output_type=output_type)

    # unshuffle returned indices and unflatten back to the correct shape
    unshuffled = tf.map_fn(
        lambda x: tf.gather(x[1], x[0]), (argminmax, shuffle_perm), fn_output_signature=output_type
    )
    unflat_shape = tf.concat([shape[:axis], shape[axis + 1 :]], 0)
    unflattened = tf.reshape(unshuffled, unflat_shape)
    return unflattened
