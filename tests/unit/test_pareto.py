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

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.data import Dataset
from trieste.utils.pareto import Pareto, non_dominated_sort


@pytest.mark.parametrize(
    "scores, pareto_set, dominance",
    [
        (
            {
                "OBJECTIVE1": Dataset(
                    tf.ones([8, 1]),
                    tf.constant(
                        [
                            [0.9575],
                            [0.9649],
                            [0.1576],
                            [0.9706],
                            [0.9572],
                            [0.4854],
                            [0.8003],
                            [0.1419],
                        ]
                    ),
                ),
                "OBJECTIVE2": Dataset(
                    tf.ones([8, 1]),
                    tf.constant(
                        [
                            [0.4218],
                            [0.9157],
                            [0.7922],
                            [0.9595],
                            [0.6557],
                            [0.0357],
                            [0.8491],
                            [0.9340],
                        ]
                    ),
                ),
            },
            tf.constant([[0.1576, 0.7922], [0.4854, 0.0357], [0.1419, 0.934]]),
            tf.constant([1, 5, 0, 7, 1, 0, 2, 0]),
        )
    ],
)
def test_dominated_sort(scores: tf.Tensor, pareto_set: tf.Tensor, dominance: tf.Tensor) -> None:
    d1, d2 = non_dominated_sort(scores)
    npt.assert_array_almost_equal(d1, pareto_set)
    npt.assert_array_almost_equal(d2, dominance)


class TestPareto:
    def setup_method(self):
        objective_scores = {
            "OBJECTIVE1": Dataset(
                tf.ones([8, 1]),
                tf.constant(
                    [[0.9575], [0.9649], [0.1576], [0.9706], [0.9572], [0.4854], [0.8003], [0.1419]]
                ),
            ),
            "OBJECTIVE2": Dataset(
                tf.ones([8, 1]),
                tf.constant(
                    [[0.4218], [0.9157], [0.7922], [0.9595], [0.6557], [0.0357], [0.8491], [0.9340]]
                ),
            ),
        }

        self.p_2d = Pareto(objective_scores)
        self.p_generic = Pareto(
            {
                "OBJECTIVE1": Dataset(tf.zeros((1, 2)), tf.zeros((1, 1))),
                "OBJECTIVE2": Dataset(tf.zeros((1, 2)), tf.zeros((1, 1))),
            }
        )
        self.p_generic.update(objective_scores, generic_strategy=True)

        scores_3d_set1 = {
            "OBJECTIVE1": Dataset(tf.ones((3, 3)), tf.constant([[2.0], [2.0], [3.0]])),
            "OBJECTIVE2": Dataset(tf.ones((3, 3)), tf.constant([[2.0], [0.0], [1.0]])),
            "OBJECTIVE3": Dataset(tf.ones((3, 3)), tf.constant([[0.0], [1.0], [0.0]])),
        }

        scores_3d_set2 = {
            "OBJECTIVE1": Dataset(tf.ones((3, 3)), tf.constant([[2.0], [2.0], [3.0]])),
            "OBJECTIVE2": Dataset(tf.ones((3, 3)), tf.constant([[0.0], [2.0], [1.0]])),
            "OBJECTIVE3": Dataset(tf.ones((3, 3)), tf.constant([[1.0], [0.0], [0.0]])),
        }

        self.p_3d_set1 = Pareto(
            {
                "OBJECTIVE1": Dataset(tf.zeros((1, 3)), tf.zeros((1, 1))),
                "OBJECTIVE2": Dataset(tf.zeros((1, 3)), tf.zeros((1, 1))),
                "OBJECTIVE3": Dataset(tf.zeros((1, 3)), tf.zeros((1, 1))),
            }
        )
        self.p_3d_set1.update(scores_3d_set1)
        self.p_3d_set2 = Pareto(
            {
                "OBJECTIVE1": Dataset(tf.zeros((1, 3)), tf.zeros((1, 1))),
                "OBJECTIVE2": Dataset(tf.zeros((1, 3)), tf.zeros((1, 1))),
                "OBJECTIVE3": Dataset(tf.zeros((1, 3)), tf.zeros((1, 1))),
            }
        )
        self.p_3d_set2.update(scores_3d_set2)

    def test_update(self):
        print(self.p_2d.bounds.lb)
        print(tf.constant([[0, 0], [1, 0], [2, 0], [3, 0]]))
        np.testing.assert_almost_equal(
            self.p_2d.bounds.lb.numpy(),
            tf.constant([[0, 0], [1, 0], [2, 0], [3, 0]]),
            err_msg="LBIDX incorrect.",
        )
        np.testing.assert_almost_equal(
            self.p_2d.bounds.ub.numpy(),
            tf.constant([[1, 4], [2, 1], [3, 2], [4, 3]]),
            err_msg="UBIDX incorrect.",
        )
        np.testing.assert_almost_equal(
            self.p_2d.front.numpy(),
            tf.constant([[0.1419, 0.9340], [0.1576, 0.7922], [0.4854, 0.0357]]),
            decimal=4,
            err_msg="PF incorrect.",
        )

        np.testing.assert_almost_equal(
            self.p_generic.bounds.lb.numpy(),
            tf.constant([[3, 0], [2, 0], [1, 2], [0, 2], [0, 0]]),
            err_msg="LBIDX incorrect.",
        )
        np.testing.assert_almost_equal(
            self.p_generic.bounds.ub.numpy(),
            tf.constant([[4, 3], [3, 2], [2, 1], [1, 4], [2, 2]]),
            err_msg="UBIDX incorrect.",
        )
        np.testing.assert_almost_equal(
            self.p_generic.front.numpy(),
            tf.constant([[0.1419, 0.9340], [0.1576, 0.7922], [0.4854, 0.0357]]),
            decimal=4,
            err_msg="PF incorrect.",
        )

        assert not np.array_equal(self.p_2d.bounds.lb.numpy(), self.p_generic.bounds.lb.numpy())
        assert not np.array_equal(self.p_2d.bounds.ub.numpy(), self.p_generic.bounds.ub.numpy())

    def test_hypervolume(self):
        np.testing.assert_almost_equal(
            self.p_2d.hypervolume([2.0, 2.0]), 3.3878, decimal=2, err_msg="hypervolume incorrect."
        )
        np.testing.assert_almost_equal(
            self.p_generic.hypervolume([2.0, 2.0]),
            3.3878,
            decimal=2,
            err_msg="hypervolume incorrect.",
        )

        # note: in original gpflowopt decimal=20 were used
        np.testing.assert_almost_equal(
            self.p_2d.hypervolume([1.0, 1.0]),
            self.p_generic.hypervolume([1.0, 1.0]),
            decimal=6,
            err_msg="hypervolume of different strategies incorrect.",
        )

        np.testing.assert_equal(
            self.p_3d_set1.hypervolume([4.0, 4.0, 4.0]).numpy(),
            29.0,
            err_msg="3D hypervolume incorrect.",
        )
        np.testing.assert_equal(
            self.p_3d_set2.hypervolume([4.0, 4.0, 4.0]).numpy(),
            29.0,
            err_msg="3D hypervolume incorrect.",
        )
