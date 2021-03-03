import tensorflow as tf
from itertools import combinations
from typing import Union
from ...utils.pareto import Pareto
from .function import HypervolumeBatchAcquisitionBuilder, get_nadir_point
from ..function import DEFAULTS, Dataset, ProbabilisticModel, \
    AcquisitionFunction, TensorType, BatchReparametrizationSampler


class BatchMonteCarloHypervolumeExpectedImprovement(HypervolumeBatchAcquisitionBuilder):
    """
    Use of the inclusion-exclusion method
    refer
    @article{daulton2020differentiable,
    title={Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization},
    author={Daulton, Samuel and Balandat, Maximilian and Bakshy, Eytan},
    journal={arXiv preprint arXiv:2006.05078},
    year={2020}
    }
    """

    def __init__(self, sample_size: int = 512, *, jitter: float = DEFAULTS.JITTER,
                 nadir_setting: Union[str, callable] = "default"):
        """
        :param sample_size: The number of samples for each batch of points.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or
            ``jitter`` is negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        super().__init__()

        self._sample_size = sample_size
        self._jitter = jitter
        self.q = -1
        self._nadir_setting = nadir_setting

    def _calculate_nadir(self, pareto: Pareto, nadir_setting="default"):
        """
        calculate the reference point for hypervolme calculation
        :param pareto: Pareto class
        :param nadir_setting
        """
        if nadir_setting == "default":
            return get_nadir_point(pareto.front)
        else:
            assert callable(nadir_setting), ValueError(
                "nadir_setting: {} do not understood".format(nadir_setting)
            )
            return nadir_setting(pareto.front)

    def __repr__(self) -> str:
        """"""
        return f"BatchMonteCarloExpectedImprovement({self._sample_size!r}, jitter={self._jitter!r})"

    def _cache_q_subset_indices(self, q: int) -> None:
        r"""Cache indices corresponding to all subsets of `q`.
        This means that consecutive calls to `forward` with the same
        `q` will not recompute the indices for all (2^q - 1) subsets.
        Note: this will use more memory than regenerating the indices
        for each i and then deleting them, but it will be faster for
        repeated evaluations (e.g. during optimization).
        Args:
            q: batch size
        """
        if q != self.q:
            indices = list(range(q))
            self.q_subset_indices = {
                f"q_choose_{i}": tf.constant(list(combinations(indices, i)))
                for i in range(1, q + 1)
            }
            self.q = q

    def prepare_acquisition_function(
            self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: The data from the observer. Must be populated.
        :param model: The model over the specified ``dataset``. Must have event shape [1].
        :return: The batch *expected improvement* acquisition function.
        :raise ValueError (or InvalidArgumentError): If ``dataset`` is not populated, or ``model``
            does not have an event shape of [1].
        """

        tf.debugging.assert_positive(len(dataset), message='Dataset must be populated.')

        means, _ = model.predict(dataset.query_points)

        datasets_mean = tf.concat(means, axis=1)
        _pf = Pareto(Dataset(query_points=tf.zeros_like(datasets_mean), observations=datasets_mean))
        _nadir_pt = self._calculate_nadir(_pf, nadir_setting=self._nadir_setting)
        lb_points, ub_points = _pf.get_partitioned_cell_bounds(_nadir_pt)
        sampler = BatchReparametrizationSampler(self._sample_size, model)

        def batch_hvei(at: TensorType) -> TensorType:
            """
            :param at: Batches of query points at which to sample the predictive distribution, with
            shape `[..., B, D]`, for batches of size `B` of points of dimension `D`. Must have a
            consistent batch size across all calls to :meth:`sample` for any given
            Complexity: O(num_obj * SK(2^q - 1))
            """
            # [..., S, B, num_obj]
            samples = sampler.sample(at, jitter=self._jitter)

            q = at.shape[-2]  # B
            self._cache_q_subset_indices(q)

            areas_per_segment = None
            # Inclusion-Exclusion loop
            for j in range(1, q + 1):
                # 选择combination
                q_choose_j = self.q_subset_indices[f"q_choose_{j}"]
                # get combination of subsets: [..., S, B, num_obj] -> [..., S, Cq_j, j, num_obj]
                obj_subsets = tf.gather(samples, q_choose_j, axis=-2)
                # get lower vertices of overlap: [..., S, Cq_j, j, num_obj] -> [..., S, Cq_j, num_obj]
                overlap_vertices = tf.reduce_max(obj_subsets, axis=-2)

                # compare overlap vertices and lower bound of each cell: -> [..., S, K, Cq_j, num_obj]
                overlap_vertices = tf.maximum(tf.expand_dims(overlap_vertices, -3),
                                              lb_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :])

                # get hvi length within each cell:-> [..., S, Cq_j, K, num_obj]
                lengths_j = tf.maximum((ub_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :]
                                        - overlap_vertices), 0.0)
                # take product over hyperrectangle side lengths to compute area within each K
                # sum over all subsets of size Cq_j #
                areas_j = tf.reduce_sum(tf.reduce_prod(lengths_j, axis=-1), axis=-1)
                # [..., S, K]
                areas_per_segment = (-1) ** (j + 1) * areas_j if areas_per_segment is None \
                    else areas_per_segment + (-1) ** (j + 1) * areas_j

            # sum over segments(cells) and average over MC samples
            # return tf.reduce_mean(batch_improvement, axis=-1, keepdims=True)  # [..., 1]
            areas_in_total = tf.reduce_sum(areas_per_segment, axis=-1)
            return tf.reduce_mean(areas_in_total, axis=-1, keepdims=True)

        return batch_hvei