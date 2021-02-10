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
""" This package contains library utilities. """
from typing_extensions import Final

from . import objectives, pareto
from .misc import Err, Ok, Result, T_co, jit, shapes_equal, to_numpy


class DEFAULTS:
    """ Default constants used in Trieste. """

    JITTER: Final[float] = 1e-6
    """
    The default jitter, typically used to stabilise computations near singular points, such as in
    Cholesky decomposition.
    """
