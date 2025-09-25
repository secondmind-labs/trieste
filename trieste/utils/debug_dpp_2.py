import gpflow
from gpflow.kernels import Exponential, Matern52
from gpflow.utilities import to_default_float

import matplotlib.pyplot as plt
import numpy as np
from gpflow.utilities.ops import square_distance
import tensorflow as tf

from trieste.data import Dataset
from trieste.models.gpflow.inducing_point_selectors import greedy_inference_dpp
from trieste.types import TensorType

PROP_OUTSIDE_RING = 0.5
N_POINTS = 1000
N_SAMPLE_POINTS = 100

def plot(ax):
    rng = np.random.default_rng(42)

    dataset = Dataset(
        query_points=to_default_float(rng.uniform(size=(N_POINTS, 2))),
        observations=to_default_float(rng.uniform(size=(N_POINTS, 1)))
    )
    dists = square_distance(dataset.query_points, to_default_float([[0.5, 0.5]]))[:, 0] ** 0.5

    ring_radius = (0.5 / np.pi) ** 0.5  # ring should have area of half the unit square

    quality_scores = tf.where(
        dists < ring_radius,
        tf.ones_like(dists),
        tf.ones_like(dists) * PROP_OUTSIDE_RING,
    ).numpy()

    dpp = greedy_inference_dpp(
        M=N_SAMPLE_POINTS,
        kernel=Matern52(variance=0.2),
        quality_scores=quality_scores,
        dataset=dataset,
    )

    dpp_dists = square_distance(dpp, to_default_float([[0.5, 0.5]]))[:, 0] ** 0.5
    n_points_outside_ring = np.sum(dpp_dists > ring_radius)

    fig, ax = plt.subplots()
    ax.scatter(dpp[:, 0].numpy(), dpp[:, 1].numpy(), color="blue")
    circle = plt.Circle((0.5, 0.5), ring_radius, color="green", fill=False)
    ax.add_patch(circle)
    ax.set_title(f"DPP Investigation")
    ax.text(0.02, 0.98, f"Points outside ring: {n_points_outside_ring}", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top')
    plt.show()


if __name__ == "__main__":
    plot()

def greedy_inference_dpp(
    M: int,
    kernel: gpflow.kernels.Kernel,
    weights: TensorType,
    dataset: Dataset,
) -> TensorType:
    """
    Get a greedy approximation of the MAP estimate of the Determinantal Point Process (DPP)
    over ``dataset`` following the algorithm of :cite:`chen2018fast`. Note that we are using the
    quality-diversity decomposition of a DPP, specifying both a similarity ``kernel``
    and ``quality_scores``.

    :param M: Desired set size.
    :param kernel: The underlying kernel of the DPP.
    :param quality_scores: The quality score of each item in ``dataset``.
    :return: The MAP estimate of the DPP.
    :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty or if the shape of
        ``quality_scores`` does not match that of ``dataset.observations``.
    """
    quality_scores = tf.ones_like(weights)
    tf.debugging.Assert(dataset is not None, [])
    tf.debugging.assert_equal(tf.shape(dataset.observations)[0], tf.shape(quality_scores)[0])
    tf.debugging.Assert(len(dataset.query_points) >= M, [])

    chosen_indicies = []  # iteratively store chosen points

    N = tf.shape(dataset.query_points)[0]
    c = tf.zeros((M - 1, N))  # [M-1,N]
    d_squared = kernel.K_diag(dataset.query_points)  # [N]

    scores = d_squared * quality_scores**2  # [N]
    chosen_indicies.append(tf.argmax(scores))  # get first element
    for m in range(M - 1):  # get remaining elements
        ix = tf.cast(chosen_indicies[-1], dtype=tf.int32)  # increment Cholesky with newest point
        newest_point = dataset.query_points[ix]

        d_temp = tf.math.sqrt(d_squared[ix])  # [1]

        L = kernel.K(dataset.query_points, newest_point[None, :])[:, 0]  # [N]
        if m == 0:
            e = L / d_temp
            c = tf.expand_dims(e, 0)  # [1,N]
        else:
            c_temp = c[:, ix : ix + 1]  # [m,1]
            e = (L - tf.matmul(tf.transpose(c_temp), c[:m])) / d_temp  # [N]
            c = tf.concat([c, e], axis=0)  # [m+1, N]
            e = tf.squeeze(e, 0)

        d_squared -= e**2
        d_squared = tf.maximum(d_squared, 1e-50)  # numerical stability

        scores = d_squared * quality_scores**2  # [N]

        remaining_indices = [x for x in range(N) if x not in chosen_indicies]
        subset_weights = tf.gather(weights, remaining_indices)
        p = subset_weights / np.sum(subset_weights)
        subset_indices = p.random.choice(range(N), size=int(len(remaining_indices) * 0.1), replace=True, p=p)
        subset_scores = tf.gather(scores, subset_indices)
        subset_max_index = tf.argmax(subset_scores)
        max_index = subset_indices[subset_max_index]

        chosen_indicies.append(max_index)  # get next element as point with largest score

    return tf.gather(dataset.query_points, chosen_indicies)
