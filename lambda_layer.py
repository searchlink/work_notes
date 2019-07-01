# -*- coding: utf-8 -*-
# @Time    : 2019/7/1 8:41
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : lambda_layer.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.backend as K

# 基于lambda层，单输入返回多个输出
x = keras.layers.Input(shape=(3,))

def f(x):
    max_x = K.max(x, axis=1)
    min_x = K.min(x, axis=1)
    return max_x, min_x

y1, y2 = keras.layers.Lambda(f)(x)
y = keras.layers.Add()([y1, y2])
model = keras.models.Model(x, y)

inp = np.array([[2, 3, 4], [5, 4, 2]])
res = model.predict(inp)

print(res)
# array([6., 7.], dtype=float32)

#############################################
# 基于lambda层，单输入返回多个输出
x1 = keras.layers.Input(shape=(3,))
x2 = keras.layers.Input(shape=(3,))
x = keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([x1, x2])
model = keras.models.Model([x1, x2], x)
inp_x1 = np.array([[2, 3, 4], [5, 4, 2]])
inp_x2 = np.array([[2, 3, 4], [5, 4, 2]]) + 1

model.predict([inp_x1, inp_x2])