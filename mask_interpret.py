# -*- coding: utf-8 -*-
# @Time    : 2019/6/27 11:21
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : mask_interpret.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.backend as K

'''https://stackoverflow.com/questions/39510809/mean-or-max-pooling-with-masking-support-in-keras'''

class MeanLayer(keras.layers.Layer):
    def __init__(self, supports_masking=True):
        super(MeanLayer, self).__init__()
        self.supports_masking = supports_masking

    def build(self, input_shape):
        self.built = True

    def call(self, x, mask=None):
        '''mask是上一层的'''
        '''# using 'mask' you can access the mask passed from the previous layer'''
        # x [batch_size, seq_len, embedding_size]
        if self.supports_masking:
            # mask [batch_size, seq_len]
            if mask is None:
                # 先判断是否非零，然后执行OR运算，计算每个序列的有效长度
                mask = K.any(K.not_equal(x, 0), -1)  # [batch_size, seq_len]
                mask = K.cast(mask, K.floatx())
                return K.sum(x, axis=1) / K.sum(mask, axis=1, keepdims=True)

            if mask is not None:
                mask = K.cast(mask, K.floatx())
                # [batch_size, embedding_size, seq_len]
                mask = K.repeat(mask, x.shape[-1].value)
                # [batch_size, seq_len, embedding_size]
                mask = tf.transpose(mask, [0, 2, 1])
                x = x * mask
                return K.sum(x, axis=1) / K.sum(mask, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

data = np.array([[[10, 10], [0, 0], [0, 0], [0, 0]],
        [[10, 10], [20, 20], [0, 0], [0, 0]],
        [[10, 10], [20, 20], [30, 30], [0, 0]],
        [[10, 10], [20, 20], [30, 30], [40, 40]]])


def build_model(use_mask_layer=True):
    inp = keras.layers.Input(shape=(4, 2))
    if use_mask_layer:
        mask_inp = keras.layers.Masking()(inp)
        out = MeanLayer(supports_masking=False)(mask_inp)
    else:
        out = MeanLayer()(inp)

    model = keras.models.Model(inputs=[inp], outputs=[out])
    return model

# 设置在上一层中调用mask层
model = build_model(use_mask_layer=True)
print(model.predict(data))

# 不在上一层中调用mask层
model = build_model(use_mask_layer=False)
print(model.predict(data))