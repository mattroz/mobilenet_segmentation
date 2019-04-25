import numpy as np

import keras
import keras.backend as K
import tensorflow as tf

from keras.layers import Conv2D, GlobalAveragePooling2D, Multiply


def SE_block(input_tensor, block_id, reduction_rate=16):
    """Channel Squeeze and Excitation block
    Paper: https://arxiv.org/pdf/1709.01507v1.pdf
    """
    if input_tensor.dtype != tf.float32:
        input_tensor = tf.cast(input_tensor, tf.float32)

    tensor_shape = K.int_shape(input_tensor)
    filters = tensor_shape[-1] if K.image_data_format() == 'channels_last' \
        else tensor_shape[1]

    over_reduction = filters // reduction_rate == 0
    reduced_filters = (filters // reduction_rate) if not over_reduction else 1

    squeeze_layer = GlobalAveragePooling2D(data_format=K.image_data_format(), name=f'squeeze_{block_id}')(input_tensor)
    squeeze_layer = K.expand_dims(K.expand_dims(squeeze_layer, axis=1), axis=1)
    excitation_1 = Conv2D(kernel_size=(1, 1),
                          filters=reduced_filters,
                          activation='relu',
                          name=f'excitate_1_{block_id}')(squeeze_layer)
    excitation_2 = Conv2D(kernel_size=(1, 1),
                          filters=filters,
                          activation='sigmoid',
                          name=f'excitate_2_{block_id}')(excitation_1)

    return Multiply(name=f'scale_{block_id}')([input_tensor, excitation_2])
