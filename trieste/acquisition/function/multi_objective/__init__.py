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
""" This folder contains multi-objective optimization utilities. """
from .dominance import non_dominated
from .pareto import Pareto, get_reference_point
from .partition import (
    DividedAndConquerNonDominated,
    ExactPartition2dNonDominated,
    prepare_default_non_dominated_partition_bounds,
)
