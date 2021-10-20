import math
from abc import ABC, abstractmethod

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.cluster.vq import kmeans

from ..types import TensorType

"""
TF compatible inducing point selection routines.

uniform_sampler: Choose points at random

k_means: Choose points as the centroids from k-mean clustering

conditional_variance: Choose points following RobustGP

GIBBON: Our new approach (BO-specific)
"""


class InducingPointSelector(ABC):
    def __init__(self, search_Space):
        self._space = search_Space

    @abstractmethod
    def get_points(
            self, X: TensorType, Y: TensorType, M: int, kernel: gpflow.kernels.Kernel, noise: float
    ):
        """
        :param X: Training data from which inducing points are to be chosen.
        :param Y: Observations at the training data.
        :param M: Desired number of inducing points.
        :param kernel: Gpflow kernel used to calulate diversity of inducing points.
        """

        raise NotImplementedError


class UniformSampler(InducingPointSelector):
    def get_points(
            self, X: TensorType, Y: TensorType, M: int, kernel: gpflow.kernels.Kernel, noise: float
    ):
        N = len(X)
        if N < M:
            raise ValueError("Need N>M")

        indicies = tf.random.categorical(tf.math.log(tf.ones([1, N])), M)
        return tf.gather(X, tf.squeeze(indicies, 0))


class RandomSampler(InducingPointSelector):
    def get_points(
            self, X: TensorType, Y: TensorType, M: int, kernel: gpflow.kernels.Kernel, noise: float
    ):
        return self._space.sample(M)


class KMeans(InducingPointSelector):
    def get_points(
            self, X: TensorType, Y: TensorType, M: int, kernel: gpflow.kernels.Kernel, noise: float
    ):

        N = len(X)
        if N < M:
            raise ValueError("Need N>M")

        perm = tf.random.shuffle(tf.range(N))
        X = tf.gather(X, perm)

        X_stds = tf.math.reduce_std(X, 0)

        if tf.math.count_nonzero(X_stds) == len(X_stds):
            X_norm = X / X_stds
        else:
            X_norm = X

        centroids, _ = kmeans(X_norm, int(M))
        if len(centroids) < M:  # sometimes scipy returns fewer centroids
            extra_points = M - len(centroids)
            extra_indicies = tf.random.categorical(tf.math.log(tf.ones([1, N])), extra_points)
            extra_centroids = tf.gather(X_norm, tf.squeeze(extra_indicies, 0))
            centroids = tf.concat([centroids, extra_centroids], axis=0)

        if tf.math.count_nonzero(X_stds) == len(X_stds):
            return centroids * X_stds
        else:
            return centroids


class ConditionalVariance(InducingPointSelector):
    def get_points(
            self, X: TensorType, Y: TensorType, M: int, kernel: gpflow.kernels.Kernel, noise: float
    ):

        N = len(X)
        if N < M:
            raise ValueError("Need N>M")

        X = X
        perm = tf.random.shuffle(tf.range(N))
        X = tf.gather(X, perm)

        chosen_indicies = []  # iteratively store chosen points

        c = tf.zeros((M - 1, N))  # [M-1,N]
        d_squared = kernel.K_diag(X) + 1e-12  # [N] jitter
        chosen_indicies.append(tf.argmax(d_squared))  # get first element

        for m in range(M - 1):  # get remaining elements
            ix = chosen_indicies[-1]  # increment Cholesky with newest point
            newest_point = X[ix]
            d_temp = tf.math.sqrt(d_squared[ix])  # [1]

            L = kernel.K(X, newest_point)  # [N]
            if m == 0:
                e = L / d_temp
                c = tf.expand_dims(e, 0)  # [1,N]
            else:
                c_temp = c[:, ix: ix + 1]  # [m,1]
                e = (L - tf.matmul(tf.transpose(c_temp), c[:m])) / d_temp  # [N]
                c = tf.concat([c, e], axis=0)  # [m+1, N]
                e = tf.squeeze(e, 0)

            d_squared -= e ** 2
            d_squared = tf.clip_by_value(d_squared, 0, math.inf)  # numerical stability

            chosen_indicies.append(tf.argmax(d_squared))

        return tf.gather(X, chosen_indicies)


class RandomConditionalVariance(InducingPointSelector):
    def get_points(
            self, X: TensorType, Y: TensorType, M: int, kernel: gpflow.kernels.Kernel, noise: float
    ):

        N = len(X)
        if N < M:
            raise ValueError("Need N>M")

        X = X
        perm = tf.random.shuffle(tf.range(N))
        X = tf.gather(X, perm)

        chosen_indicies = []  # iteratively store chosen points

        c = tf.zeros((M - 1, N))  # [M-1,N]
        d_squared = kernel.K_diag(X) + 1e-12  # [N] jitter
        choice = tf.random.categorical(tf.math.log(tf.expand_dims(d_squared, 0)), 1)
        chosen_indicies.append(choice[0][0])  # get first element

        for m in range(M - 1):  # get remaining elements
            ix = chosen_indicies[-1]  # increment Cholesky with newest point
            newest_point = X[ix]
            d_temp = tf.math.sqrt(d_squared[ix])  # [1]

            L = kernel.K(X, newest_point)  # [N]
            if m == 0:
                e = L / d_temp
                c = tf.expand_dims(e, 0)  # [1,N]
            else:
                c_temp = c[:, ix: ix + 1]  # [m,1]
                e = (L - tf.matmul(tf.transpose(c_temp), c[:m])) / d_temp  # [N]
                c = tf.concat([c, e], axis=0)  # [m+1, N]
                e = tf.squeeze(e, 0)

            d_squared -= e ** 2
            d_squared = tf.clip_by_value(d_squared, 0, math.inf)  # numerical stability

            choice = tf.random.categorical(tf.math.log(tf.expand_dims(d_squared, 0)), 1)
            chosen_indicies.append(choice[0][0])  # get next element

        return tf.gather(X, chosen_indicies)


class GIBBON(InducingPointSelector):
    def get_points(
            self, X: TensorType, Y: TensorType, M: int, kernel: gpflow.kernels.Kernel, noise: float
    ):

        N = len(X)
        if N < M:
            raise ValueError("Need N>M")

        perm = tf.random.shuffle(tf.range(N))
        X = tf.gather(X, perm)
        Y = tf.gather(Y, perm)

        chosen_indicies = []  # iteratively store chosen points

        c = tf.zeros((M - 1, N))  # [M-1,N]
        K = kernel.K_diag(X) + noise  # [N] jitter

        # estimate mutual information (FOR NOW WE ASSUME EXACT EVALS)
        eta = tf.reduce_mean(Y)
        gamma = (eta - tf.squeeze(Y, 1)) / tf.math.sqrt(K)  # [N]
        normal = tfp.distributions.Normal(tf.cast(0, Y.dtype), tf.cast(1, Y.dtype))
        minus_cdf = 1 - normal.cdf(gamma)
        minus_cdf = tf.clip_by_value(minus_cdf, 1.0e-10, 1)  # clip below to improve numer
        MI = -gamma * normal.prob(gamma) / (2 * minus_cdf) - tf.math.log(minus_cdf)
        q = (1 / tf.math.sqrt(K)) * tf.math.exp(MI)
        d_squared = tf.math.exp(MI) ** 2
        chosen_indicies.append(tf.argmax(d_squared))  # get first element

        for m in range(M - 1):  # get remaining elements
            ix = chosen_indicies[-1]  # increment Cholesky with newest point
            newest_point = X[ix]
            d_temp = tf.math.sqrt(d_squared[ix])  # [1]
            K = kernel.K(X, newest_point)  # [N]
            K = K + tf.where(
                tf.equal(tf.range(N), tf.cast(ix, dtype=tf.int32)),
                noise * tf.ones(N, dtype=tf.float64),
                tf.zeros(N, dtype=tf.float64),
            )
            L = q * K * q[ix]

            if m == 0:
                e = L / d_temp
                c = tf.expand_dims(e, 0)  # [1,N]
            else:
                c_temp = c[:, ix: ix + 1]  # [m,1]
                e = (L - tf.matmul(tf.transpose(c_temp), c[:m])) / d_temp  # [N]
                c = tf.concat([c, e], axis=0)  # [m+1, N]
                e = tf.squeeze(e, 0)

            d_squared -= e ** 2
            d_squared = tf.clip_by_value(d_squared, 0, math.inf)  # numerical stability

            chosen_indicies.append(tf.argmax(d_squared))  # get next element

        return tf.gather(X, chosen_indicies)
