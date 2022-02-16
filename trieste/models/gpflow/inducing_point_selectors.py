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
This module is the home of  Trieste's functionality for controlling the inducing points
of sparse variational Gaussian processes.
"""


from __future__ import annotations

import tensorflow as tf

import math
from abc import ABC, abstractmethod

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.cluster.vq import kmeans

from ...data import Dataset
from ...models import ProbabilisticModel
from ...types import TensorType
from ...space import Box, SearchSpace

class InducingPointSelector(ABC):
    def __init__(self, search_space: SearchSpace, model : ProbabilisticModel):
        self._search_space = search_space
        self._model = model
        # todo add required checks


    @abstractmethod
    def get_points(self, M: int, dataset: Optional[Dataset]=None):
        """
        :param M: Desired number of inducing points.
        :param dataset: TODO
        """
        raise NotImplementedError


class UniformSampler(InducingPointSelector):
	"""
	TODO
	"""

    @abstractmethod
    def get_points(self, M: int, dataset: Optional[Dataset]=None):
        """
        :param M: Desired number of inducing points. NOT OPTIONAL
        :param dataset: TODO
        """

        N = tf.shape(dataset.query_points)[0]
        if N < M: 
            raise ValueError("Need N>M") # TODO make select random for rest?

        indicies = tf.random.categorical(tf.math.log(tf.ones([1, N])), M)
        return tf.gather(X, tf.squeeze(indicies, 0))


class RandomSampler(InducingPointSelector):
	"""
	TODO
	"""
    def get_points(self, M: int, dataset: Optional[Dataset]=None):
    	
    	if isinstance(self.search_space, Box):
        	return self._space.sample_halton(M)
        else:
            return self._space.sample(M)	


class KMeans(InducingPointSelector):
    """
	TODO
	"""  

    def get_points(self, M: int, dataset: Optional[Dataset]=None):
    
        N = len(dataset.query_points)
        if N < M:
            raise ValueError("Need N>M")

        perm = tf.random.shuffle(tf.range(N))
        
        X = tf.gather(dataset.query_points, perm)
        X_stds = tf.math.reduce_std(X, 0)
        if tf.math.count_nonzero(X_stds)==len(X_stds):
            X_norm = X / X_stds
        else:
            X_norm = X

        centroids, _ = kmeans(X_norm, int(M))
        if len(centroids) < M:  # sometimes scipy returns fewer centroids
            extra_points = M - len(centroids)
            extra_indicies = tf.random.categorical(tf.math.log(tf.ones([1, N])), extra_points)
            extra_centroids = tf.gather(X_norm, tf.squeeze(extra_indicies, 0))
            centroids = tf.concat([centroids, extra_centroids], axis=0)
        
        if tf.math.count_nonzero(X_stds)==len(X_stds):
            return centroids * X_stds
        else:
            return centroids