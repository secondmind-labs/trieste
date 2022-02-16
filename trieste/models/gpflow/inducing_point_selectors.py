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
This module is the home of  Trieste's functionality for choosing the inducing points
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
    """
    TODO
    """

    def __init__(self, search_space: SearchSpace, model : ProbabilisticModel, resample_every_step : bool = True):
        self._search_space = search_space
        self._model = model
        self._resample_for_every_step = resample_every_step
        # todo add required checks


    @abstractmethod
    def get_inducing_points(self, M: int, dataset: Optional[Dataset]=None):
        """
        :param M: Desired number of inducing points.
        :param dataset: TODO 
        :return: TODO perhaps need Variable
        """
        raise NotImplementedError

    def update_inducing_points(self, inducing_points: TensorType, dataset: Optional[Dataset]=None):
        """
        TODO

        :param inducing_points: Current inducing point locations (to be updated by calling this method).
        :param dataset: TODO
        """
        M = tf.shape(inducing_points)[0]

        if resample_every_step: # get M new points
            return inducing_points.update(self.get_inducing_points(M, dataset))
        else: # otherwise keep the same points
            return inducing_points








class UniformSampler(InducingPointSelector):
	"""
	TODO
	"""

    @abstractmethod
    def get_points(self,inducing_points: TensorType, dataset: Optional[Dataset]=None):
        """
        :param inducing_points: Current inducing point locations (to be updated by calling this method).
        :param dataset: TODO (not optional)
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
    def get_points(self, inducing_points: TensorType, dataset: Optional[Dataset]=None):
    	
    	if isinstance(self.search_space, Box):
        	return self._space.sample_halton(M)
        else:
            return self._space.sample(M)	


class KMeans(InducingPointSelector):
    """
	TODO
	"""  

    def get_points(self, inducing_points: TensorType, dataset: Optional[Dataset]=None):
    
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