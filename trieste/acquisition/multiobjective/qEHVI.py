import tensorflow as tf
from itertools import combinations
from typing import Mapping
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

    def __init__(self, sample_size: int = 512, *, jitter: float = DEFAULTS.JITTER):
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

    # TODO: Use of state for pareto update?
    def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: [str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        """
        :param datasets: The data from the observer. Must be populated.
        :param models: The model over the specified ``dataset``. Must have event shape [1].
        :return: The batch *expected improvement* acquisition function.
        :raise ValueError (or InvalidArgumentError): If ``dataset`` is not populated, or ``model``
            does not have an event shape of [1].
        """

        for _, data in datasets.items():
            tf.debugging.assert_positive(len(data))

        means = [models[model_tag].predict(datasets[data_tag].query_points)[0] for
                 model_tag, data_tag in zip(models, datasets)]
        for mean in means:
            tf.debugging.assert_shapes(
                [(mean, ["_", 1])], message="Expected model with event shape [1]."
            )
        datasets_mean = tf.concat(means, axis=1)
        pareto = Pareto(Dataset(query_points=tf.zeros_like(datasets_mean), observations=datasets_mean))
        lb_points, ub_points = pareto.get_partitioned_cell_bounds(get_nadir_point(pareto.front))
        samplers = [BatchReparametrizationSampler(self._sample_size, models[tag]) for tag in models]

        def batch_hvei(at: TensorType) -> TensorType:
            """
            :param at: Batches of query points at which to sample the predictive distribution, with
            shape `[..., B, D]`, for batches of size `B` of points of dimension `D`. Must have a
            consistent batch size across all calls to :meth:`sample` for any given
            Complexity: O(num_obj * SK(2^q - 1))
            """
            # [..., S, B, num_obj]
            samples = tf.concat([sampler.sample(at, jitter=self._jitter) for sampler in samplers], axis=-1)

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
