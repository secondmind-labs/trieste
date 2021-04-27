from scipy.cluster.vq import kmeans
import math
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow

from abc import ABC, abstractmethod
from ..type import TensorType
"""
TF compatible inducing point selection routines.

uniform_sampler: Choose points at random

k_means: Choose points as the centroids from k-mean clustering

conditional_variance: Choose points following RobustGP

GIBBON: Our new approach (BO-specific)
"""

class InducingPointSelector(ABC):
    def __init__(self, X: TensorType, Y:TensorType, M:int, kernel: gpflow.kernels.Kernel):
        """
        :param X: Training data from which inducing points are to be chosen.
        :param Y: Observations at the training data.
        :param M: Desired number of inducing points.
        :param kernel: Gpflow kernel used to calulate diversity of inducing points.
        """
        self._X = X
        self._M = M
        self._N = len(X)
        if self._N<self._M:
            raise ValueError("Need N>M")
        self._Y = Y
        self._kernel = kernel
        

    @abstractmethod
    def get_points(self,seed:int=None):
        raise NotImplementedError


class UniformSampler(InducingPointSelector):

    def get_points(self,seed:int=None):
        if seed:
            tf.random.set_seed(seed)
        indicies = tf.random.categorical(tf.math.log(tf.ones([1,self._N]) ), self._M)
        return  tf.gather(self._X,tf.squeeze(indicies,0))


class KMeans(InducingPointSelector):

    def get_points(self,seed:int=None):
        if seed:
            tf.random.set_seed(seed)
        perm = tf.random.shuffle(tf.range(self._N))
        X = tf.gather(self._X,perm)
        
        X_stds = tf.math.reduce_std(X,0)
        X_norm = X / X_stds 
        centroids, _ = kmeans(X_norm,self._M)
        if len(centroids)<self._M: # sometimes scipy returns fewer centroids
            extra_points = self._M - len(centroids)
            extra_indicies = tf.random.categorical(tf.math.log(tf.ones([1,self._N]) ), extra_points)
            extra_centroids = tf.gather(X_norm, tf.squeeze(extra_indicies,0))
            centroids = tf.concat([centroids, extra_centroids], axis=0)
        return centroids * X_stds



class ConditionalVariance(InducingPointSelector):

    def get_points(self,seed:int=None):
        if seed:
            tf.random.set_seed(seed)
        perm = tf.random.shuffle(tf.range(self._N))
        X = tf.gather(self._X,perm)
        
        chosen_indicies = [] # iteratively store chosen points
        
        c = tf.zeros((self._M-1,self._N)) # [M-1,N]
        d_squared = self._kernel.K_diag(X) + 1e-12 # [N] jitter
        chosen_indicies.append(tf.argmax(d_squared)) # get first element
        
        for m in range(self._M-1): # get remaining elements
            ix = chosen_indicies[-1] # increment Cholesky with newest point
            newest_point = X[ix]
            d_temp =  tf.math.sqrt(d_squared[ix]) # [1]
            
            L = self._kernel.K(X, newest_point) # [N] 
            if m==0:
                e = L/d_temp
                c = tf.expand_dims(e,0) # [1,N]
            else:
                c_temp = c[:,ix:ix+1] # [m,1]
                e = (L - tf.matmul(tf.transpose(c_temp),c[:m])) / d_temp # [N] 
                c = tf.concat([c,e],axis=0) # [m+1, N]
                e = tf.squeeze(e,0)
            
            d_squared -= e ** 2
            d_squared = tf.clip_by_value(d_squared,0,math.inf) # numerical stability

            chosen_indicies.append(tf.argmax(d_squared))
      
        return  tf.gather(X,chosen_indicies)


class GIBBON(InducingPointSelector):

    def get_points(self,seed:int=None):
        if seed:
            tf.random.set_seed(seed)

        perm = tf.random.shuffle(tf.range(self._N))
        X = tf.gather(self._X,perm)
        Y = tf.gather(self._Y,perm)
        
        chosen_indicies = [] # iteratively store chosen points
        
        c = tf.zeros((self._M-1,self._N)) # [M-1,N]
        K = self._kernel.K_diag(X) + 1e-12 # [N] jitter
        
        # estimate mutual information (FOR NOW WE ASSUME EXACT EVALS)
        eta = tf.reduce_mean(Y)
        gamma = (eta - tf.squeeze(Y,1)) / tf.math.sqrt(K) # [N]
        normal = tfp.distributions.Normal(tf.cast(0, Y.dtype), tf.cast(1, Y.dtype))
        minus_cdf = 1 - normal.cdf(gamma)
        minus_cdf = tf.clip_by_value(minus_cdf, 1.0e-10, 1)  # clip below to improve numer
        MI =  -gamma * normal.prob(gamma) / (2 * minus_cdf) - tf.math.log(minus_cdf)
        q =(1 / tf.math.sqrt(K)) * tf.math.exp(MI)
        d_squared = tf.math.exp(MI)**2
        
        chosen_indicies.append(tf.argmax(d_squared)) # get first element
        
        for m in range(self._M-1): # get remaining elements
            ix = chosen_indicies[-1] # increment Cholesky with newest point
            newest_point = X[ix]
            d_temp =  tf.math.sqrt(d_squared[ix]) # [1]
            K = self._kernel.K(X, newest_point) # [N] 
            L = q * K * q[ix]
            
            if m==0:
                e = L/d_temp
                c = tf.expand_dims(e,0) # [1,N]
            else:
                c_temp = c[:,ix:ix+1] # [m,1]
                e = (L - tf.matmul(tf.transpose(c_temp),c[:m])) / d_temp # [N] 
                c = tf.concat([c,e],axis=0) # [m+1, N]
                e = tf.squeeze(e,0)
            
            d_squared -= e ** 2
            d_squared = tf.clip_by_value(d_squared,0,math.inf) # numerical stability

                
            chosen_indicies.append(tf.argmax(d_squared)) # get next element
      
        return  tf.gather(X,chosen_indicies)    
