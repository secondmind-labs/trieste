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
"""
The library root. See :mod:`~trieste.bayesian_optimizer` for the core optimizer, which requires
models (see :mod:`~trieste.models`), and data sets (see :mod:`~trieste.data`). The
:mod:`~trieste.acquisition` package provides a selection of acquisition algorithms and the
functionality to define your own. The :mod:`~trieste.ask_tell_optimization` package provides API
for Ask-Tell optimization and manual control of the optimization loop.
The :mod:`~trieste.objectives` package contains several popular objective functions,
useful for experimentation.
"""
from . import (
    acquisition,
    ask_tell_optimization,
    bayesian_optimizer,
    data,
    models,
    objectives,
    observer,
    space,
    types,
    utils,
)
from .version import __version__
