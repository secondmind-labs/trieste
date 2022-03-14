import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from trieste.objectives import (
    michalewicz_5,
    ackley_5,
    branin
)

from trieste.objectives.utils import mk_observer
import trieste
from trieste.acquisition.rule import DiscreteThompsonSampling
