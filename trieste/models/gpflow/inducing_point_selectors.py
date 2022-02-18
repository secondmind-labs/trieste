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
from ..interfaces import ProbabilisticModel
from ...types import TensorType
from ...space import Box, SearchSpace

from .builders import NUM_INDUCING_POINTS_PER_DIM, MAX_NUM_INDUCING_POINTS



class InducingPointSelector(ABC):
    """
    TODO
    """

 
    @abstractmethod
    def update(self, current_inducing_points: TensorType, model: Optional[ProbabilisticModel], dataset: Optional[Dataset]=None):
        """
        TODO
        """
        


class DummyInducingPointSelector(InducingPointSelector):
    """
    Does nuffing
    """
    def update(self, current_inducing_points: TensorType, model: Optional[ProbabilisticModel], dataset: Optional[Dataset]=None):
        """
        TODO
        """
        return current_inducing_points



class DynamicInducingPointSelector(ABC):
    """
    TODO
    """

    def __init__(self, search_space: SearchSpace,recalc_every_model_update : bool = True):
        """
        TODO
        """ 

        self._search_space = search_space
        self._recalc_every_model_update = recalc_every_model_update
        self._initialized = False
        # todo add required checks


    
    def update(self, current_inducing_points: TensorType, model: Optional[ProbabilisticModel], dataset: Optional[Dataset]=None):
        """
        TODO
        """
        if isinstance(self.model.inducing_variable, SeparateIndependentInducingVariables):
            raise ValueError(
                f"TODO"
            )     

        if self._initialized and (not self._recalc_every_model_update): # TODO
            return current_inducing_points
        else:
            self._initialized = True
            M = tf.shape(current_inducing_points)[0]

            # TODO check new is of same shape!

            return self._recalculate_inducing_points(M, model, dataset)
 

    @abstractmethod
    def _recalculate_inducing_points(M: int, model: Optional[ProbabilisticModel], dataset: Optional[Dataset]=None):
        """
        :param M: Desired number of inducing points.
        :param model: TODO
        :param dataset: TODO 
        :return: TODO
        """
        raise NotImplementedError




class UniformInducingPointSelector(DynamicInducingPointSelector):
    """
    Choose points at random across space
    """
    def _recalculate_inducing_points(self,M: int, model: Optional[ProbabilisticModel], dataset: Optional[Dataset]=None):
        
        if isinstance(self._search_space, Box):
            return self._search_space.sample_sobol(M)
        else:
            return self._search_space.sample(M) 



class RandomSubSampleInducingPointSelector(DynamicInducingPointSelector):
    """
    Choose points at random from training data (fill remainder uniformly)
    """

    def _recalculate_inducing_points(self,M: int, model: Optional[ProbabilisticModel], dataset: Optional[Dataset]=None):
        """
        :param inducing_points: Current inducing point locations (to be updated by calling this method).
        :param dataset: TODO (not optional)
        """
        # TODO say need dataset


        N = tf.shape(dataset.query_points)[0] # training data size
        random_indicies = tf.random.categorical(tf.math.log(tf.ones([1, N])), tf.math.minimum(N, M))
        sub_sample = tf.gather(X, tf.squeeze(random_indicies, 0)) # [min(N, M), d]

        if N < M: # if fewer data than inducing points then sample remaining uniformly
            uniform_sampler =  UniformInducingPointSelector(self._search_space, self._model)
            uniform_sample = uniform_sampler.get_inducing_points(M-N) # [M-N, d]
           
        return tf.concat([sub_sample, uniform_sample],0) # [M, d]


class KMeansInducingPointSelector(DynamicInducingPointSelector):
    """
	Choose points as the centroids from k-mean clustering
	"""  

    def _recalculate_inducing_points(self,M: int, model: Optional[ProbabilisticModel], dataset: Optional[Dataset]=None):
    
        # TODO say need dataset

        query_points = dataset.query_points # [N, d]
        N = tf.shape(query_points)[0] 

        shuffled_query_points = tf.shuffle(query_points) # [N, d]
        query_points_stds = tf.math.reduce_std(shuffled_query_points, 0)  # [d]

        if  tf.math.count_nonzero(X_stds) == N:
            normalize = True
            shuffled_query_points = shuffled_query_points / query_points_stds # [N, d]
        else:
            normalize = False

        centroids, _ = kmeans(shuffled_query_points, int(tf.math.minimum(M,N))) # [C, d]

        if len(centroids) < M:  # sometimes scipy returns fewer centroids or if less data than desred TODO so choose rest as uniform
            uniform_sampler =  UniformInducingPointSelector(self._search_space, self._model)
            extra_centroids = uniform_sampler.get_inducing_points(M - len(centroids)) # [M-C, d]
            extra_centroids = extra_centroids / query_points_stds # remember to standardize
            centroids = tf.concat([centroids, extra_centroids], axis=0) # [M, d]
        
        if normalize:
            return centroids * query_points_stds # [M, d]
        else:
            return centroids # [M, d]
