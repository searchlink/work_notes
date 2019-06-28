# -*- coding: utf-8 -*-
# @Time    : 2019/6/28 11:05
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : share_layer.py
# @Software: PyCharm

'''
keras中关于共享layer的权重：
参考如下链接：
https://stackoverflow.com/questions/39564579/keras-reuse-weights-from-a-previous-layer-converting-to-keras-tensor
https://keras.io/getting-started/functional-api-guide/#shared-layers
https://stackoverflow.com/questions/49800201/reusing-a-group-of-keras-layers
https://keras.io/getting-started/functional-api-guide/#shared-vision-model
'''

import tensorflow as tf
import tensorflow.python.keras as keras

def share_weights(hidden_units=128):
    '''
    reuse a group of keras layers (共享多层)
    '''
    layers_units = (80, 40, 1)
    share_input = keras.layers.Input(shape=(hidden_units*2, ))
    share_layer = share_input
    for i in range(len(layers_units)):
        activation = None if i == 2 else "sigmoid"
        share_layer = keras.layers.Dense(layers_units[i], activation=activation)(share_layer)
    out_layer = share_layer
    model = keras.models.Model(share_input, out_layer)
    return model

inp = tf.random_normal((64, 256))
model = share_weights()
print(model(inp))

'''共享单层'''
hidden_units = 128
share_fcn = keras.layers.Dense(hidden_units)  # 定义共享某一层的weights

# 定义输入
a = keras.layers.Input(shape=(280, 256))
b = keras.layers.Input(shape=(280, 256))

fcn_a = share_fcn(a)
fcn_b = share_fcn(b)
print(fcn_a)
print(fcn_b)