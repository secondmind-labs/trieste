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

from typing import TypeVar

import gpflow
import tensorflow as tf
from gpflow.base import default_float
from gpflux.layers import KG, GPLayer, LatentVariableLayer
from gpflux.layers.inverse_wishart_layer import InverseWishart
from gpflux.models import DeepGP, DeepIWP
from gpflux.sampling.sample import Sample

from ...types import TensorType

M = TypeVar("M", bound=tf.Module)
""" A type variable bound to :class:`tf.Module`. """


def sample_consistent_lv_layer(layer: LatentVariableLayer) -> Sample:
    class SampleLV(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            sample = layer.prior.sample()
            batch_shape = tf.shape(X)[:-1]
            sample_rank = tf.rank(sample)
            for _ in range(len(batch_shape)):
                sample = tf.expand_dims(sample, 0)
            sample = tf.tile(
                sample, tf.concat([batch_shape, tf.ones(sample_rank, dtype="int32")], -1)
            )
            return layer.compositor([X, sample])

    return SampleLV()


def sample_dgp(model: DeepGP) -> Sample:
    function_draws = []
    for layer in model.f_layers:
        if isinstance(layer, GPLayer):
            function_draws.append(layer.sample())
        elif isinstance(layer, LatentVariableLayer):
            function_draws.append(sample_consistent_lv_layer(layer))
        else:
            raise NotImplementedError(f"Sampling not implemented for {type(layer)}")

    class ChainedSample(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            for f in function_draws:
                X = f(X)
            return X

    return ChainedSample()


# AKA inefficient sampling
def sample_diwp(model: DeepIWP) -> Sample:
    class DIWPSample(Sample):
        Gii_list = [
            tf.Variable(tf.ones([1, 0, 0], dtype=tf.float64), shape=[1, None, None])
            for _ in range(len(model.f_layers) - 1)
        ]

        u = tf.ones([1, 0, 0, 0], dtype=tf.float64)

        X = None
        Git_old = [None for _ in range(len(model.f_layers) - 1)]
        Gtt_old = [None for _ in range(len(model.f_layers) - 1)]
        f_old = tf.zeros([1, 0, 1], dtype=default_float())
        N_old = 0

        def __call__(self, X_new: TensorType) -> tf.Tensor:
            Gii_is_populated = tf.not_equal(tf.size(self.Gii_list[0]), 0)

            X_new = tf.expand_dims(X_new, 0)
            N_new = tf.shape(X_new)[1]

            if self.X is None:
                self.X = X_new
            else:
                self.X = tf.concat([self.X, X_new], axis=1)

            inputs = model._inducing_add(self.X)
            x = model.kernelizer(inputs, full_cov=True)

            for i, layer in enumerate(model.f_layers[:-1]):

                K = layer.kernel_gram(x, full_cov=True)

                dKii = layer.delta * K.ii
                dKit = layer.delta * K.it
                dKtt = layer.delta * K.tt

                dKtt_new = dKtt[:, self.N_old :, self.N_old :]

                full_dK = tf.concat(
                    [tf.concat([dKii, dKit], 2), tf.concat([tf.linalg.adjoint(dKit), dKtt], 2)], 1
                )

                dKit_new = full_dK[:, : layer.P + self.N_old, layer.P + self.N_old :]
                dKii_new = full_dK[:, : layer.P + self.N_old, : layer.P + self.N_old]

                chol_dKii_new = tf.linalg.cholesky(
                    dKii_new
                    + (
                        gpflow.default_jitter()
                        * tf.reduce_max(dKii_new)
                        * tf.eye(tf.shape(dKii_new)[-1], dtype=gpflow.default_float())
                    )
                )

                if not Gii_is_populated:
                    Gii, _ = layer.Gii(dKii)
                    self.Gii_list[i].assign(Gii)

                if self.Git_old[i] is None:
                    Gii = self.Gii_list[i]
                else:
                    Gii = tf.concat(
                        [
                            tf.concat([self.Gii_list[i], self.Git_old[i]], 2),
                            tf.concat([tf.linalg.adjoint(self.Git_old[i]), self.Gtt_old[i]], 2),
                        ],
                        1,
                    )

                inv_Kii_kit = tf.linalg.cholesky_solve(chol_dKii_new, dKit_new)

                dKtti = dKtt_new - tf.linalg.matmul(dKit_new, inv_Kii_kit, transpose_a=True)

                nu = (
                    layer.delta
                    + tf.cast(tf.shape(dKii_new)[-1], gpflow.default_float())
                    + tf.cast(tf.shape(dKtt_new)[-1], gpflow.default_float())
                    + 1
                )
                Ptti = InverseWishart(dKtti, nu)
                Gtti_sample = Ptti.rsample([])
                Gtti = Gtti_sample + (
                    gpflow.default_jitter()
                    * tf.reduce_max(Gtti_sample)
                    * tf.eye(tf.shape(Gtti_sample)[-1], dtype=gpflow.default_float())
                )

                inv_Gii_git = inv_Kii_kit + tf.linalg.matmul(
                    tf.linalg.triangular_solve(
                        tf.linalg.adjoint(chol_dKii_new),
                        tf.random.normal(tf.shape(dKit_new), dtype=gpflow.default_float()),
                        lower=False,
                    ),
                    tf.linalg.cholesky(Gtti),
                    transpose_b=True,
                )
                Git = Gii @ inv_Gii_git

                Gtt = Gtti + tf.linalg.matmul(Git, inv_Gii_git, transpose_a=True)

                if self.Git_old[i] is None:
                    self.Git_old[i] = Git[..., : layer.P, :]
                    self.Gtt_old[i] = Gtt
                else:
                    self.Git_old[i] = tf.concat([self.Git_old[i], Git[..., : layer.P, :]], 2)
                    self.Gtt_old[i] = tf.concat(
                        [
                            tf.concat([self.Gtt_old[i], Git[..., layer.P :, :]], 2),
                            tf.concat([tf.linalg.adjoint(Git[..., layer.P :, :]), Gtt], 2),
                        ],
                        1,
                    )

                x = KG(self.Gii_list[i], self.Git_old[i], self.Gtt_old[i])

            layer = model.f_layers[-1]

            x = layer.kernel_gram(x, full_cov=True)

            u_is_populated = tf.not_equal(tf.size(self.u), 0)

            Kuu = x.ii
            Kuf = x.it
            Kfu = tf.linalg.adjoint(Kuf)
            Kff = x.tt

            Kuu, Kuf, Kfu = tf.expand_dims(Kuu, 1), tf.expand_dims(Kuf, 1), tf.expand_dims(Kfu, 1)

            if not u_is_populated:
                u, _, _ = layer.sample_u(Kuu)
                self.u = u

            full_dK = tf.concat(
                [
                    tf.concat([Kuu, Kuf], 3),
                    tf.concat([tf.linalg.adjoint(Kuf), tf.expand_dims(Kff, 1)], 3),
                ],
                2,
            )

            Kuu = full_dK[..., : model.num_inducing + self.N_old, : model.num_inducing + self.N_old]
            Kuf = full_dK[..., : model.num_inducing + self.N_old, model.num_inducing + self.N_old :]
            Kff = tf.squeeze(
                full_dK[..., model.num_inducing + self.N_old :, model.num_inducing + self.N_old :],
                1,
            )

            chol_Kuu = tf.linalg.cholesky(
                Kuu
                + (
                    gpflow.default_jitter()
                    * tf.reduce_max(Kuu)
                    * tf.eye(tf.shape(Kuu)[-1], dtype=gpflow.default_float())
                )
            )

            Kfu_invKuu = tf.linalg.adjoint(tf.linalg.cholesky_solve(chol_Kuu, Kuf))
            u_full = tf.concat([self.u, tf.expand_dims(tf.linalg.adjoint(self.f_old), -1)], 2)
            Ef = tf.linalg.adjoint(tf.squeeze(Kfu_invKuu @ u_full, -1))

            Vf = Kff - tf.squeeze(Kfu_invKuu @ Kuf, 1)
            Vf = Vf + (
                gpflow.default_jitter()
                * tf.reduce_max(Vf)
                * tf.eye(tf.shape(Vf)[-1], dtype=gpflow.default_float())
            )

            f = Ef + tf.linalg.cholesky(Vf) @ tf.random.normal(
                tf.shape(Ef), dtype=gpflow.default_float()
            )

            self.f_old = tf.concat([self.f_old, f], 1)

            self.N_old = self.N_old + N_new

            if not isinstance(layer.mean_function, gpflow.mean_functions.Zero):
                f = f + layer.mean_function.c

            return tf.squeeze(f, 0)

    return DIWPSample()
