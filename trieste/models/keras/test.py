#%%
import tensorflow as tf
from layers import DropConnect
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras import Input
from tensorflow.keras.models import Model

# %%
tf.random.set_seed(1234)
x = tf.constant([[1, 5, 7], [1, 4, 7]])
d = DropConnect(units=3, p_dropout=0)
y = tf.constant([[3., -4., 9.], [5., 1., 6.]])
dense = tf.keras.layers.Dense(3)
#%%

dc1 = DropConnect(p_dropout = 1, units = 3,)
dc0 = DropConnect(p_dropout = 0, units = 3,)

inputs = Input(shape=(3,))
m1 = Model(inputs=inputs, outputs=dc1(inputs))
m0 = Model(inputs=inputs, outputs=dc0(inputs))

m0.compile(Adam(), MeanAbsoluteError())
m1.compile(Adam(), MeanAbsoluteError())

h0 = m0.fit(x, y)
h1 = m1.fit(x, y)

zero_weights = tf.zeros(shape=dc1.get_weights()[0].shape)
zero_bias = tf.zeros(shape=dc1.get_weights()[0].shape)
one_weights = tf.ones(shape=dc1.get_weights()[1].shape)
one_bias = tf.ones(shape=dc1.get_weights()[1].shape)

dense = tf.keras.layers.Dense(3)
m = Model(inputs, dense(inputs))
m.set_weights([zero_weights, zero_bias])
m.compile(Adam(), MeanAbsoluteError())
h = m.fit(x, y)
# %%
