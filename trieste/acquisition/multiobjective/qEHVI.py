from trieste.acquisition.function import AcquisitionFunctionBuilder, SingleModelBatchAcquisitionBuilder, DEFAULTS, \
    Dataset, ProbabilisticModel, AcquisitionFunction, TensorType, BatchReparametrizationSampler
import tensorflow as tf
from typing import Mapping
from trieste.utils.pareto import Pareto
from abc import ABC, abstractmethod
from trieste.acquisition.function import BatchAcquisitionFunctionBuilder
from math import inf
from itertools import combinations


class MultiModelBatchAcquisitionBuilder(ABC):
    """
    Convenience acquisition function builder for a batch acquisition function (or component of a
    composite batch acquisition function) that requires only one model, dataset pair.
    """

    def using(self, tags: [str]) -> BatchAcquisitionFunctionBuilder:
        """
        :param tag: The tag for the model, dataset pair to use to build this acquisition function.
        :return: A batch acquisition function builder that selects the model and dataset specified
            by ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """
        multi_builder = self

        # 这个一点用都没有: 我们本来就是要使用multi-model的
        class _Anon(BatchAcquisitionFunctionBuilder):
            def prepare_acquisition_function(
                    self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
            ) -> AcquisitionFunction:
                # TODO: seems can create subdicts instead of using the original one
                return multi_builder.prepare_acquisition_function(datasets, models)

            def __repr__(self) -> str:
                return f"{multi_builder!r} using tag {tags!r}"

        return _Anon()

    @abstractmethod
    def prepare_acquisition_function(
            self, dataset: Mapping[str, Dataset], model: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        """
        :param dataset: The data to use to build the acquisition function.
        :param model: The model over the specified ``dataset``.
        :return: An acquisition function.
        """


class BatchMultiModelReparametrizationSampler:
    r"""
    FIXME: Note this currently assume each model has only ONE output dim!
     otherwise the output dim will get concatenated with different model output

    This sampler employs the *reparameterization trick* to approximate batches of samples from a
    :class:`ProbabilisticModel`\ 's predictive joint distribution as
    .. math:: x \mapsto \mu(x) + \epsilon L(x)
    where :math:`L` is the Cholesky factor s.t. :math:`LL^T` is the covariance, and
    :math:`\epsilon \sim \mathcal N (0, 1)` is constant for a given sampler, thus ensuring samples
    form a continuous curve.
    """

    def __init__(self, sample_size: int, models: Mapping[str, ProbabilisticModel]):
        """
        :param sample_size: The number of samples for each batch of points. Must be positive.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)

        self._sample_size = sample_size

        # _eps is essentially a lazy constant. It is declared and assigned an empty tensor here, and
        # populated on the first call to sample,

        self._eps = tf.Variable(
            tf.ones([0, 0, sample_size], dtype=tf.float64), shape=[None, None, sample_size]
            # shape: [predictive dimension, Batchsize, Samplesize]
        )  # [0, 0, S]
        self._models = models

    def __repr__(self) -> str:
        """"""
        return f"BatchReparametrizationSampler({self._sample_size!r}, {self._models!r})"

    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        math:
        ζ~f
        f~N(μ, Cov)
        ζ = μ + L*Ɛ


        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`BatchReparametrizationSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`BatchReparametrizationSampler` instances will produce different samples.
        :param at: Batches of query points at which to sample the predictive distribution, with
            shape `[..., B, num_query_points, D]`, for batches of size `B` of points of dimension `D`. Must have a
            consistent batch size across all calls to :meth:`sample` for any given
            :class:`BatchReparametrizationSampler`.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :return: The samples, of shape `[..., S, B, L]`, where `S` is the `sample_size`, `B` the
            number of points per batch, and `L` the dimension of the model's predictive
            distribution.
        :raise ValueError (or InvalidArgumentError): If any of the following are true:
            - ``at`` is a scalar.
            - The batch size `B` of ``at`` is not positive.
            - The batch size `B` of ``at`` differs from that of previous calls.
            - ``jitter`` is negative.
        """

        tf.debugging.assert_rank_at_least(at, 2)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        batch_size = at.shape[-2]

        tf.debugging.assert_positive(batch_size)

        eps_is_populated = tf.size(self._eps) != 0

        # eps shape:
        if eps_is_populated:
            tf.debugging.assert_equal(
                batch_size,
                tf.shape(self._eps)[-2],
                f"{type(self).__name__} requires a fixed batch size. Got batch size {batch_size}"
                f" but previous batch size was {tf.shape(self._eps)[-2]}.",
            )
        # mean, cov = [self._model.predict_joint(at)]  # [..., B, L], [..., L, B, B]
        predicts = [self._models[model_tag].predict_joint(at) for model_tag in self._models]
        means, covs = list(zip(*predicts))
        means = tf.concat(means, axis=-1)  # [..., B, num_obj]
        # 2021/2/22 12:55 fixed axi
        # covs = tf.concat(covs, axis=1)  # [num_obj, B, B]
        covs = tf.concat(covs, axis=-3)  # [num_obj, B, B]
        # candidate_mean, candidate_var = (tf.concat(moment, 1) for moment in zip(*predicts))

        # Consider wether this need to be done through different obj: yes, refer independent_reparameterization sampler in trieste
        if not eps_is_populated:
            self._eps.assign(
                tf.random.normal(
                    # [mean.shape[-1], batch_size, self._sample_size], dtype=tf.float64
                    [means.shape[-1], batch_size, self._sample_size], dtype=tf.float64
                )  # Note: Current shape: [num_obj, B, S] instead of [num_obj, L, B, S]
            )

        # Cov+jitter * identity = LL^T
        identity = tf.eye(batch_size, dtype=covs.dtype)  # [B, B]
        covs_cholesky = tf.linalg.cholesky(covs + jitter * identity)  # [..., num_obj, B, B]

        # matrix multiplication

        # # [120, 2, 3, 3] @ [2, 3, 1000] = [120, 2, 3, 1000],
        #  这个乘法的进行为 (120, 2)[3, 3] @ (2) [3, 1000], 真正的单元乘法就是[3, 3]@[3, 1000]
        variances_contribution = covs_cholesky @ tf.cast(self._eps, covs.dtype)  # [..., num_obj, B, S]

        # shape transform: [..., num_obj, B, S] -> [..., B, S, num_obj]
        leading_indices = tf.range(tf.rank(variances_contribution) - 3)
        absolute_trailing_indices = [-1, -2, -3] + tf.rank(variances_contribution)
        new_order = tf.concat([leading_indices, absolute_trailing_indices], axis=0)
        #
        return means[..., None, :, :] + tf.transpose(variances_contribution, new_order)


#  TODO: Testing
# FIXME: Not working properly when use small traning data to init
class BatchMonteCarloHypervolumeExpectedImprovement(MultiModelBatchAcquisitionBuilder):
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

    def __init__(self, sample_size: int, *, jitter: float = DEFAULTS.JITTER, q=1):
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
        # self.q = -1
        self.q = q

    def __repr__(self) -> str:
        """"""
        return f"BatchMonteCarloExpectedImprovement({self._sample_size!r}, jitter={self._jitter!r})"

        # here

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

    def _get_q_dicts(self) -> dict:
        # 2020/2/22 New Ver:

        indices = list(range(self.q))
        self.q_subset_indices = {
            f"q_choose_{i}": tf.constant(list(combinations(indices, i)))
            for i in range(1, self.q + 1)
        }
        return self.q_subset_indices

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

        # datasets_mean = tf.concat([models[model_tag].predict(datasets[data_tag].query_points)[0]
        #                            for model_tag, data_tag in zip(models, datasets)], axis=1)
        means = [models[model_tag].predict(datasets[data_tag].query_points)[0] for
                 model_tag, data_tag in zip(models, datasets)]
        for mean in means:
            tf.debugging.assert_shapes(
                [(mean, ["_", 1])], message="Expected model with event shape [1]."
            )
        datasets_mean = tf.concat(means, axis=1)
        pareto = Pareto(Dataset(query_points=tf.zeros_like(datasets_mean), observations=datasets_mean))
        nadir_point = get_nadir_point(pareto.front)

        sampler = BatchMultiModelReparametrizationSampler(self._sample_size, models)
        q_choose_j_dict = self._get_q_dicts()

        def batch_hvei(at: TensorType) -> TensorType:
            samples = sampler.sample(at, jitter=self._jitter)  # [..., S, B, num_obj]
            q = at.shape[-2]  # parallel point, aka, B
            # FIXME: For debugging, we manually specify q at the beginning of acq, this should be extracted from
            #  somewhere else later
            # self._cache_q_subset_indices(q)
            # N = tf.shape(samples)[0]
            # num_samples = tf.shape(samples)[1]
            outdim = tf.shape(samples)[-1]
            num_cells = tf.shape(pareto.bounds.lb)[0]
            # Inclusion Exclusion Principle (i.e., Eq. 5)
            pf_ext = tf.concat(
                [
                    -inf * tf.ones([1, outdim], dtype=pareto.front.dtype),
                    pareto.front,
                    nadir_point,
                ],
                0,
            )
            col_idx = tf.tile(tf.range(outdim), (num_cells,))
            ub_idx = tf.stack((tf.reshape(pareto.bounds.ub, [-1]), col_idx), axis=1)  # 上界索引
            lb_idx = tf.stack((tf.reshape(pareto.bounds.lb, [-1]), col_idx), axis=1)  # 下界索引

            ub_points = tf.reshape(tf.gather_nd(pf_ext, ub_idx), [num_cells, outdim])
            lb_points = tf.reshape(tf.gather_nd(pf_ext, lb_idx), [num_cells, outdim])
            #
            # areas_per_segment = tf.zeros(shape=(*tf.shape(samples)[:2], num_cells), dtype=samples.dtype)
            areas_per_segment = None
            for j in range(1, q + 1):
                # 选择combination
                # 2020/2/22 为debug，把这里移出去
                # q_choose_j = self.q_subset_indices[f"q_choose_{j}"]
                q_choose_j = q_choose_j_dict[f"q_choose_{j}"]
                # 根据combination索引, shape是对的，element是对的
                # samples: [120, 1000, 3, 2] -> [120, 1000, Cn_j, j, 2]
                obj_subsets = tf.gather(samples, q_choose_j, axis=-2)
                # samples: [120, 1000, Cn_j, j, 2] -> [120, 1000, Cn_j, 2]
                # FIXME? Reduce max??
                overlap_vertices = tf.reduce_max(obj_subsets, axis=-2)
                # overlap_vertices = tf.reduce_min(obj_subsets, axis=-2)
                # overlap_vertices = obj_subsets.min(dim=-2).values
                # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
                # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m

                # 这个应该是对每个cell，比较upper bounds, 这两行其实是照搬botorch改的，理解不是很到位
                # TODO: compare [120, 1000, Cn_j, 2] and [m, 2]
                overlap_vertices = tf.maximum(tf.expand_dims(overlap_vertices, -3),
                                              lb_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :])
                # tf.reshape(lb_points, [1, 1, lb_points.shape[0], 1, lb_points.shape[-1]]))
                # substract cell lower bounds, clamp min at zero
                # tf.reshape(ub_points, [1, 1, ub_points.shape[0], 1, ub_points.shape[-1]])
                lengths_j = tf.maximum((ub_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :]
                                        - overlap_vertices), 0.0)
                # take product over hyperrectangle side lengths to compute area
                # sum over all subsets of size i # TODO:
                areas_j = tf.reduce_sum(tf.reduce_prod(lengths_j, axis=-1), axis=-1)
                # areas_j = areas_j, axis=-1)
                areas_per_segment = (-1) ** (j + 1) * areas_j if areas_per_segment is None \
                    else areas_per_segment + (-1) ** (j + 1) * areas_j

            # sum over segments(cells) and average over MC samples
            # return tf.reduce_mean(batch_improvement, axis=-1, keepdims=True)  # [..., 1]
            areas_in_total = tf.reduce_sum(areas_per_segment, axis=-1)
            return tf.reduce_mean(areas_in_total, axis=-1, keepdims=True)

        # debug use
        # return lambda at: batch_hvei(at)
        return batch_hvei


def get_nadir_point(front: tf.Tensor) -> tf.Tensor:
    """
    nadir point calculation method
    """
    f = tf.math.reduce_max(front, axis=0, keepdims=True) - tf.math.reduce_min(
        front, axis=0, keepdims=True
    )
    return tf.math.reduce_max(front, axis=0, keepdims=True) + 2 * f / front.shape[0]