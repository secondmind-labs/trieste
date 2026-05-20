# Copyright (C) Secondmind Ltd 2026 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
from __future__ import annotations

from functools import partial
from math import pi
from pathlib import Path

import numpy as np
import tensorflow as tf

from trieste.objectives.single_objectives import michalewicz
from trieste.objectives.utils import mk_observer
from trieste.space import Box
from trieste.types import TensorType

NUM_POINTS = 100_000
TEST_FRACTION = 0.2
RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent

search_space = Box(
    tf.constant([0.0], dtype=tf.float32),
    tf.constant([pi], dtype=tf.float32),
) ** 20
observer = mk_observer(partial(michalewicz, d=20))


def train_test_split(
    query_points: TensorType,
    observations: TensorType,
    test_fraction: float,
) -> tuple[tuple[TensorType, TensorType], tuple[TensorType, TensorType]]:
    """
    Shuffle query points and observations, then split into train and test sets.

    :param query_points: Input locations, shape ``[n, d]``.
    :param observations: Target values, shape ``[n, 1]`` (or matching leading dims).
    :param test_fraction: Fraction of rows assigned to the test set.
    :return: ``((train_qp, train_obs), (test_qp, test_obs))``.
    """
    n = int(query_points.shape[0])
    indices = tf.random.shuffle(tf.range(n))
    shuffled_qp = tf.gather(query_points, indices)
    shuffled_obs = tf.gather(observations, indices)

    n_test = int(n * test_fraction)
    test_qp, train_qp = shuffled_qp[:n_test], shuffled_qp[n_test:]
    test_obs, train_obs = shuffled_obs[:n_test], shuffled_obs[n_test:]
    return (train_qp, train_obs), (test_qp, test_obs)


def write_npz(path: Path, query_points: TensorType, observations: TensorType) -> None:
    """
    Write query points and observations to a compressed NumPy archive.

    Arrays are stored as ``X`` (query points, shape ``[n, d]``) and ``y`` (observations,
    shape ``[n]``).

    :param path: Output file path (``.npz``).
    :param query_points: Input locations, shape ``[n, d]``.
    :param observations: Target values, shape ``[n, 1]`` (or broadcastable to one column).
    """
    np.savez_compressed(
        path,
        X=query_points.numpy(),
        y=observations.numpy().reshape(-1),
    )


def main() -> None:
    """Generate data, split into train/test, and write compressed NPZ files."""
    tf.random.set_seed(RANDOM_SEED)

    points = search_space.sample(NUM_POINTS)
    dataset = observer(points)

    (train_qp, train_obs), (test_qp, test_obs) = train_test_split(
        dataset.query_points,
        dataset.observations,
        TEST_FRACTION,
    )

    write_npz(OUTPUT_DIR / "train.npz", train_qp, train_obs)
    write_npz(OUTPUT_DIR / "test.npz", test_qp, test_obs)


if __name__ == "__main__":
    main()
