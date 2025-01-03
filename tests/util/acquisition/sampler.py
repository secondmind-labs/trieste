import tensorflow as tf

from tests.util.models.gpflow.models import QuadraticMeanAndRBFKernel
from trieste.models.gpflow import BatchReparametrizationSampler
from trieste.types import TensorType
from trieste.utils import DEFAULTS


class PseudoBatchReparametrizationSampler(BatchReparametrizationSampler):
    """A Sampler that return the specified sample as deterministic samples`."""

    def __init__(self, samples: TensorType):
        """
        :param samples `[S, B, L]`, where `S` is the `sample_size`, `B` the
            number of points per batch, and `L` the dimension of the model's predictive
            distribution.
        """
        tf.debugging.assert_shapes(
            [(samples, ["S", "B", "L"])],
            message="This sampler takes samples of shape "
            "[sample_size, batch_points, output_dimension].",
        )
        self._samples = samples  # [S, B, L]
        super().__init__(1, QuadraticMeanAndRBFKernel())

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._samples!r})"

    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        :param at: Batches of query points at which to sample the predictive distribution, with
            shape `[..., B, D]`, for batches of size `B` of points of dimension `D`.
        :param jitter: unused
        :return: The samples, of shape `[..., S, B, L]`, where `S` is the `sample_size`, `B` the
            number of points per batch, and `L` the dimension of the model's predictive
            distribution.
        """

        tf.debugging.assert_rank_at_least(at, 2)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        batch_size = at.shape[-2]

        tf.debugging.assert_positive(batch_size)
        tf.assert_equal(batch_size, self._samples.shape[-2])  # assert B is equivalent
        return tf.broadcast_to(self._samples, [*at.shape[:-2], *self._samples.shape])
