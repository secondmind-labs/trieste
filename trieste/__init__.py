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
models (see :mod:`~trieste.models`), and data sets (see :mod:`~trieste.datasets`). The
:mod:`~trieste.acquisition` package provides a selection of acquisition algorithms and the
functionality to define your own.
"""
from . import (
    acquisition,
    models,
    utils,
    bayesian_optimizer,
    datasets,
    observer,
    space,
    type,
)
