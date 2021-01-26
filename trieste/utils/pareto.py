import numpy as np
import tensorflow as tf
import gpflow
from trieste.data import Dataset
from trieste.type import TensorType
from typing import Mapping

jitter = gpflow.config.default_jitter()

def atleast2d(tensor:tf.Tensor)->tf.Tensor:
    if tf.rank(tensor) <=1:
        return tf.expand_dims(tensor,axis=0)
    return tensor

def to_Dataset(datasets: Mapping[str, Dataset])->Dataset:

    data_query_points = []
    data_observations = []
    [(data_query_points.append(datasets[tag].query_points), data_observations.append(datasets[tag].observations))
        for tag in datasets]

    assert tf.math.reduce_all([tf.math.equal(input_data, data_query_points[0]) for input_data
                                in data_query_points]), ValueError('Input data is not the same for each objective')

    mo_dataset = Dataset(data_query_points[0], tf.concat(data_observations, axis=1))

    return mo_dataset


def non_dominated_sort(datasets: Mapping[str, Dataset])-> [TensorType,TensorType]:
    """
    Computes the non-dominated set for a set of data points
    :param dataset: data points
    :return: tuple of the non-dominated set and the degree of dominance,
        dominances gives the number of dominating points for each data point
    """
    dataset = to_Dataset(datasets)
    observations = dataset.observations
    extended = tf.tile(tf.expand_dims(observations,0), [observations.shape[0], 1, 1])
    swapped_ext = tf.einsum("ij...->ji...", extended)
    dominance = tf.reduce_sum(tf.cast(tf.logical_and(tf.reduce_all(extended <= swapped_ext, axis=2),
                              tf.reduce_any(extended < swapped_ext, axis=2)),tf.float32), axis=1)

    return tf.boolean_mask(observations,dominance == 0), dominance


class BoundedVolumes():

    def __init__(self, lb: tf.Tensor, ub: tf.Tensor):
        """
        Construct bounded volumes.
        :param lb: the lowerbounds of the volumes
        :param ub: the upperbounds of the volumes
        """

        assert tf.reduce_all(lb.shape == ub.shape)
        self.lb = lb
        self.ub = ub

    def append(self, lb: tf.Tensor, ub: tf.Tensor):
        """
        Add new bounded volumes.
        :param lb: the lowerbounds of the volumes
        :param ub: the upperbounds of the volumes
        """
        self.lb = tf.concat([self.lb, lb],axis=0)
        self.ub = tf.concat([self.ub, ub],axis=0)

    def clear(self):
        """
        Clears all stored bounded volumes
        """
        dtype = self.lb.dtype
        outdim = self.lb.shape[1]
        self.lb = tf.zeros([0, outdim], dtype=dtype)
        self.ub = tf.zeros([0, outdim], dtype=dtype)

    def size(self)->tf.Tensor:
        """
        :return: volume of each bounded volume
        """
        return tf.math.reduce_prod(self.ub - self.lb, axis=0, keepdims=True)

class Pareto():

    def __init__(self, datasets: Mapping[str, Dataset], threshold: tf.Tensor=0):
        """
        Construct a Pareto set.
        Stores a Pareto set and calculates the cell bounds covering the non-dominated region.
        The latter is needed for certain multiobjective acquisition functions.
        E.g., the :class:`~.acquisition.HVProbabilityOfImprovement`.
        :param y: output data points, size N x R
        :param threshold: approximation threshold for the generic divide and conquer strategy
            (default 0: exact calculation)
        """
        self.threshold = threshold
        self.data = to_Dataset(datasets)
        
        # Setup data structures
        self.bounds = BoundedVolumes(tf.zeros([0, self.data.observations.shape[1]],dtype=tf.int32),
                                     tf.zeros([0, self.data.observations.shape[1]],dtype=tf.int32))

        self.front = tf.zeros([0, self.data.observations.shape[1]])
        # Initialize
        self.update()

    @staticmethod
    def _is_test_required(smaller: tf.Tensor)-> tf.Tensor:
        """
        Tests if a point augments or dominates the Pareto set.
        :param smaller: a boolean ndarray storing test point < Pareto front
        :return: True if the test point dominates or augments the Pareto front (boolean)
        """
        # if and only if the test point is at least in one dimension smaller for every point in the Pareto set
        idx_dom_augm = tf.reduce_any(smaller, axis=1)
        is_dom_augm = tf.reduce_all(idx_dom_augm)

        return is_dom_augm

    def _update_front(self) -> tf.Tensor:
        """
        Calculate the non-dominated set of points based on the latest data.
        The stored Pareto set is sorted on the first objective in ascending order.
        :return: boolean, whether the Pareto set has actually changed since the last iteration
        """
        current = self.front
        pf, _ = non_dominated_sort(self.data)
        self.front = tf.gather_nd(pf,tf.expand_dims(tf.argsort(pf[:,0]),axis=1))
        return not np.array_equal(current.numpy(), self.front.numpy())

    def update(self, datasets: Mapping[str, Dataset], generic_strategy=False):
        """
        Update with new output data.
        Computes the Pareto set and if it has changed recalculates the cell bounds covering the non-dominated region.
        For the latter, a direct algorithm is used for two objectives, otherwise a
        generic divide and conquer strategy is employed.
        :param y: output data points
        :param generic_strategy: Force the generic divide and conquer strategy regardless of the number of objectives
            (default False)
        """

        data = to_Dataset(datasets)
        self.data = data if data is not None else self.data

        # Find (new) set of non-dominated points
        changed = self._update_front()

        # Recompute cell bounds if required
        # Note: if the Pareto set is based on model predictions it will almost always change in between optimizations
        if changed:
            # Clear data container
            self.bounds.clear()
            if generic_strategy:
                self.divide_conquer_nd()
            else:
                self.bounds_2d() if self.data.observations.shape[1] == 2 else self.divide_conquer_nd()    

    def divide_conquer_nd(self):
        """
        Divide and conquer strategy to compute the cells covering the non-dominated region.
        Generic version: works for an arbitrary number of objectives.
        """
        outdim = self.data.observations.shape[1]

        # The divide and conquer algorithm operates on a pseudo Pareto set
        # that is a mapping of the real Pareto set to discrete values
        pseudo_pf = tf.argsort(self.front, axis=0) + 1  # +1 as index zero is reserved for the ideal point

        # Extend front with the ideal and anti-ideal point
        min_pf = tf.reduce_min(self.front, axis=0) - 1
        max_pf = tf.reduce_max(self.front, axis=0) + 1

        pf_ext = tf.concat([atleast2d(min_pf), atleast2d(self.front), atleast2d(max_pf)],axis=0)# Needed for early stopping check (threshold)
        pf_ext_idx = tf.concat([atleast2d(tf.zeros(outdim, dtype=tf.int32)),
                                atleast2d(pseudo_pf),
                                atleast2d(tf.ones(outdim, dtype=tf.int32) * self.front.shape[0] + 1)],axis=0)

        # Start with one cell covering the whole front
        dc = [(tf.zeros(outdim, dtype=tf.int32),
               (int(pf_ext_idx.shape[0]) - 1) * tf.ones(outdim, dtype=tf.int32))]
        total_size = tf.reduce_prod(max_pf - min_pf)

        # Start divide and conquer until we processed all cells
        while dc:
            # Process test cell
            cell = dc.pop()
            arr = tf.range(outdim)
            idx_lb = tf.gather_nd(pf_ext_idx, tf.stack((cell[0] , arr), -1))
            idx_ub = tf.gather_nd(pf_ext_idx, tf.stack((cell[1] , arr), -1))
            lb = tf.gather_nd(pf_ext, tf.stack((idx_lb , arr), -1))
            ub = tf.gather_nd(pf_ext, tf.stack((idx_ub , arr), -1))

            # Acceptance test:
            if self._is_test_required((ub - jitter) < self.front):
                # Cell is a valid integral bound: store
                self.bounds.append(atleast2d(idx_lb), atleast2d(idx_ub))
            # Reject test:
            elif self._is_test_required((lb + jitter) < self.front):
                # Cell can not be discarded: calculate the size of the cell
                dc_dist = cell[1] - cell[0]
                hc = BoundedVolumes(lb, ub)

                # Only divide when it is not an unit cell and the volume is above the approx. threshold
                if tf.reduce_any(dc_dist > 1) and tf.reduce_all((hc.size()[0] / total_size) > self.threshold):
                    # Divide the test cell over its largest dimension
                    edge_size, idx = tf.reduce_max(dc_dist), tf.argmax(dc_dist)
                    edge_size1 = int(tf.round(tf.cast(edge_size,dtype=tf.float32) / 2.0))
                    edge_size2 = edge_size - edge_size1

                    # Store divided cells
                    ub = tf.identity(cell[1])

                    ub = tf.unstack(ub)
                    ub[idx] = ub[idx]- edge_size1
                    ub = tf.stack(ub)
                    dc.append((tf.identity(cell[0]), ub))


                    lb = tf.identity(cell[0])
                    lb = tf.unstack(lb)
                    lb[idx] = lb[idx]+ edge_size2
                    lb = tf.stack(lb)
                    dc.append((lb, tf.identity(cell[1])))
            # else: cell can be discarded

    def bounds_2d(self):
        """
        Computes the cells covering the non-dominated region for the specific case of only two objectives.
        Assumes the Pareto set has been sorted in ascending order on the first objective.
        This implies the second objective is sorted in descending order.
        """
        outdim = self.data.observations.shape[1]
        assert outdim == 2

        pf_idx = tf.argsort(self.front, axis=0)
        pf_ext_idx = tf.concat((atleast2d(tf.zeros(outdim, dtype=tf.int32)), 
                                atleast2d(pf_idx + 1), 
                                atleast2d(tf.ones(outdim,dtype=tf.int32) * self.front.shape[0] + 1)),axis=0)

        for i in range(pf_ext_idx[-1, 0]):
            self.bounds.append(atleast2d(tf.constant([i, 0],dtype=tf.int32)),
                               atleast2d(tf.constant([i+1, pf_ext_idx[-i-1, 1].numpy()],dtype=tf.int32)))

    def hypervolume(self, reference):
        """
        Autoflow method to calculate the hypervolume indicator
        The hypervolume indicator is the volume of the dominated region.
        :param reference: reference point to use
            Should be equal or bigger than the anti-ideal point of the Pareto set
            For comparing results across runs the same reference point must be used
        :return: hypervolume indicator (the higher the better)
        """

        min_pf = tf.reduce_min(self.front, 0, keepdims=True)
        R = tf.expand_dims(reference, 0)
        pseudo_pf = tf.concat((min_pf, self.front, R), 0)
        D = tf.shape(pseudo_pf)[1]
        N = tf.shape(self.bounds.ub)[0]

        idx = tf.tile(tf.expand_dims(tf.range(D), -1),[1, N])
        ub_idx = tf.reshape(tf.stack([tf.transpose(self.bounds.ub), idx], axis=2), [N * D, 2])
        lb_idx = tf.reshape(tf.stack([tf.transpose(self.bounds.lb), idx], axis=2), [N * D, 2])
        ub = tf.reshape(tf.gather_nd(pseudo_pf, ub_idx), [D, N])
        lb = tf.reshape(tf.gather_nd(pseudo_pf, lb_idx), [D, N])
        hv = tf.reduce_sum(tf.reduce_prod(ub - lb, 0))
        return tf.reduce_prod(R - min_pf) - hv