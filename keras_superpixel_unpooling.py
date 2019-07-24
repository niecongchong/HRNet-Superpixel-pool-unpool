import keras.backend as K
import tensorflow as tf
from keras.layers import Layer
import numpy as np


class SuperpixelUnpooling(Layer):
    def __init__(self, num_superpixels=100, **kwargs):
        super(SuperpixelUnpooling, self).__init__(**kwargs)
        self.num_superpixels = num_superpixels

    def compute_output_shape(self, input_shapes):
        pooled_feature_map_shape = input_shapes[0]
        superpixel_map_shape = input_shapes[1]
        return (superpixel_map_shape[0], superpixel_map_shape[1], superpixel_map_shape[2], pooled_feature_map_shape[2])

    def call(self, inputs):
        pooled_feature_map = inputs[0]  # b x k x c
        superpixel_map = tf.cast(inputs[1], tf.int32)  # b x h x w

        pooled_feature_map_shape = pooled_feature_map.get_shape().as_list()
        superpixel_map_shape = superpixel_map.get_shape().as_list()

        flat_superpixel_map_size = np.prod(superpixel_map_shape)

        # ----------------------------- gather_nd ----------------------------- #
        gather_input = pooled_feature_map

        gather_index = tf.reshape(superpixel_map, shape=[flat_superpixel_map_size, 1])
        batch_index = tf.reshape(tf.range(superpixel_map_shape[0], dtype=tf.int32),
                                 shape=[superpixel_map_shape[0], 1, 1])
        b_base = tf.ones((superpixel_map_shape[0], superpixel_map_shape[1], superpixel_map_shape[2]),
                         dtype=tf.int32) * batch_index
        batch_index = tf.reshape(b_base, [superpixel_map_shape[0]*superpixel_map_shape[1]*superpixel_map_shape[2], 1])

        gather_index = tf.concat([batch_index, gather_index], axis=-1)

        update_feature = tf.gather_nd(gather_input, gather_index)
        # ----------------------------- gather_nd ----------------------------- #

        # ----------------------------- scatter_nd ----------------------------- #
        unpooled_feature_map_shape = self.compute_output_shape([pooled_feature_map_shape, superpixel_map_shape])
        flat_unpooled_feature_map_shape = [unpooled_feature_map_shape[0],
                                           unpooled_feature_map_shape[1]*unpooled_feature_map_shape[2],
                                           unpooled_feature_map_shape[3]]

        hw_index = tf.reshape(tf.range(flat_unpooled_feature_map_shape[1], dtype=tf.int32),
                              shape=[1, flat_unpooled_feature_map_shape[1], 1])
        hw_base = tf.ones((flat_unpooled_feature_map_shape[0], flat_unpooled_feature_map_shape[1], 1),
                          dtype=tf.int32) * hw_index
        hw_index = tf.reshape(hw_base, [flat_unpooled_feature_map_shape[0]*flat_unpooled_feature_map_shape[1], 1])

        scatter_index = tf.concat([batch_index, hw_index], axis=-1)

        unpooled_feature = tf.scatter_nd(scatter_index, update_feature, shape=flat_unpooled_feature_map_shape)
        unpooled_feature = tf.reshape(unpooled_feature, unpooled_feature_map_shape)
        # ----------------------------- scatter_nd ----------------------------- #

        return unpooled_feature


# pooled_feature_map = np.random.random((2, 5, 1))
# superpixel_map = np.random.randint(0, 5, (2, 4, 4))
# print(pooled_feature_map)
# print(superpixel_map)
# pooled_feature_map = K.variable(value=pooled_feature_map)
# superpixel_map = K.variable(value=superpixel_map, dtype=tf.int32)
#
# a = SuperpixelUnpooling(num_superpixels=5)([pooled_feature_map, superpixel_map])
#
# print(a)
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(a))
