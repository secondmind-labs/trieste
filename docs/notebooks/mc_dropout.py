#%%
from distutils.command.build import build
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras import Input
from tensorflow.keras.models import Model


from tests.util.misc import empty_dataset
from trieste.models.keras import get_tensor_spec_from_data
from trieste.models.keras.architectures import DropConnectNetwork, DropoutNetwork
from trieste.models.keras.builders import build_vanilla_keras_mcdropout
from trieste.models.keras.layers import DropConnect

#%%

sample_data = empty_dataset([1], [1])
input, output = get_tensor_spec_from_data(sample_data)

dn = DropoutNetwork(input, output)

dcn = DropConnectNetwork(input, output)

dnn = build_vanilla_keras_mcdropout(sample_data)
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

zero_weights = np.zeros(shape=dc1.get_weights()[0].shape)
zero_bias = np.zeros(shape=dc1.get_weights()[1].shape)
one_weights = np.ones(shape=dc1.get_weights()[0].shape)
one_bias = np.ones(shape=dc1.get_weights()[1].shape)

dense = tf.keras.layers.Dense(3)
m = Model(inputs, dense(inputs))
m.compile(Adam(), MeanAbsoluteError())
m.set_weights([one_weights, zero_bias])
h = m.fit(x, y)
# %%
