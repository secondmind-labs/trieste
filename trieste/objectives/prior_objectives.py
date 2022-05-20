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
This module contains toy objective functions, useful for experimentation. A number of them have been
taken from `this Virtual Library of Simulation Experiments
<https://web.archive.org/web/20211015101644/https://www.sfu.ca/~ssurjano/> (:cite:`ssurjano2021`)`_.
"""

from __future__ import annotations

import pickle
from os import path
from typing import Callable, Tuple

import pandas as pd
import tensorflow as tf

from ..models.gpflux import DeepGaussianProcess, build_vanilla_deep_gp
from ..models.gpflux.sampler import DeepGaussianProcessPriorTrajectorySampler
from ..objectives.utils import mk_observer
from ..space import Box
from ..types import TensorType
from .single_objectives import ackley_2, ackley_5, michalewicz_2, michalewicz_5

function_dict = {
    "ackley_2": [ackley_2, Box([-2.0], [3.0]) ** 2],
    "ackley_5": [ackley_5, Box([-1.0], [2.0]) ** 5],
    "mich_2": [michalewicz_2, Box([-1.0], [4.0]) ** 2],
    "mich_5": [michalewicz_5, Box([-1.0], [4.0]) ** 5],
}


def build_dgp_prior_function(name: str) -> Callable[[TensorType], TensorType]:
    assert name in ["ackley_2", "ackley_5", "mich_2", "mich_5"]

    function = function_dict[name][0]
    search_space = function_dict[name][1]

    observer = mk_observer(function)

    initial_data = observer(search_space.sample_sobol(100))

    dgp = build_vanilla_deep_gp(initial_data, search_space, num_layers=2, num_inducing_points=100)
    model = DeepGaussianProcess(dgp)

    prior_sampler = DeepGaussianProcessPriorTrajectorySampler(model, num_features=1000)

    basepath = path.dirname(__file__)
    with open(path.join(basepath, "prior_weights/dgp_{}.pickle".format(name)), "rb") as f:
        loaded_state = pickle.load(f)

    prior_sampler.load_weights(loaded_state)

    return lambda x: prior_sampler.get_trajectory()(x[..., None, :])


DGP_MICH_2_SEARCH_SPACE = function_dict["mich_2"][1]

DGP_MICH_5_SEARCH_SPACE = function_dict["mich_5"][1]

DGP_ACKLEY_2_SEARCH_SPACE = function_dict["ackley_2"][1]

DGP_ACKLEY_5_SEARCH_SPACE = function_dict["ackley_5"][1]


def get_minimum_and_minimizer(name: str) -> Tuple[TensorType, TensorType]:
    basepath = path.dirname(__file__)
    minimum_path = path.join(basepath, "prior_weights/min_value_{}".format(name))
    minimizer_path = path.join(basepath, "prior_weights/minimizer_{}".format(name))

    f_min = tf.constant(pd.read_csv(minimum_path, index_col=0).values[0], tf.float64)
    f_minimizer = tf.constant(pd.read_csv(minimizer_path, index_col=0).values.T, tf.float64)

    return f_min, f_minimizer


DGP_MICH_2_MINIMUM, DGP_MICH_2_MINIMIZER = get_minimum_and_minimizer("mich_2")

DGP_MICH_5_MINIMUM, DGP_MICH_5_MINIMIZER = get_minimum_and_minimizer("mich_5")

DGP_ACKLEY_2_MINIMUM, DGP_ACKLEY_2_MINIMIZER = get_minimum_and_minimizer("ackley_2")

DGP_ACKLEY_5_MINIMUM, DGP_ACKLEY_5_MINIMIZER = get_minimum_and_minimizer("ackley_5")
