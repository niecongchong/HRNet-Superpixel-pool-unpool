import keras.backend as K
import tensorflow as tf
from keras.layers import Layer
import numpy as np


class SuperpixelPooling(Layer):
    def __init__(self, num_superpixels=100, **kwargs):
        super(SuperpixelPooling, self).__init__(**kwargs)
        self.num_superpixels = num_superpixels

    def compute_output_shape(self, input_shapes):
        feature_map_shape = input_shapes[0]
        return (feature_map_shape[0], self.num_superpixels, feature_map_shape[3])

    def call(self, inputs):
        feature_map = inputs[0]  # b x h x w x c
        superpixel_map = tf.cast(inputs[1], tf.int32)  # b x h x w

        feature_map_shape = feature_map.get_shape().as_list()
        superpixel_map_shape = superpixel_map.get_shape().as_list()

        flat_feature_map_shape = [-1, feature_map_shape[3]]
        flat_superpixel_map_size = np.prod(superpixel_map_shape)

        # ----------------------------- partition ----------------------------- #
        partition_feature = K.reshape(feature_map, shape=flat_feature_map_shape)

        partition_index = K.reshape(superpixel_map, shape=[flat_superpixel_map_size])
        batch_range = tf.reshape(tf.range(superpixel_map_shape[0], dtype=tf.int32),
                                 shape=[superpixel_map_shape[0], 1, 1])
        b = tf.ones(superpixel_map_shape, dtype=tf.int32) * batch_range
        b = tf.reshape(b, [flat_superpixel_map_size])
        partition_index = b * self.num_superpixels + partition_index

        partition_num = feature_map_shape[0] * self.num_superpixels

        group_feature = tf.dynamic_partition(partition_feature, partition_index, partition_num)
        group_feature = [tf.reduce_mean(group_feature[i], axis=0) for i in range(len(group_feature))]

        # ----------------------------- partition ----------------------------- #

        # ----------------------------- scatter_nd ----------------------------- #
        pooled_map_shape = self.compute_output_shape([feature_map_shape, superpixel_map_shape])

        batch_index = tf.reshape(tf.range(pooled_map_shape[0], dtype=tf.int32),
                                 shape=[pooled_map_shape[0], 1, 1])
        b_base = tf.ones((pooled_map_shape[0], pooled_map_shape[1], 1), dtype=tf.int32) * batch_index
        batch_index = tf.reshape(b_base, [pooled_map_shape[0]*pooled_map_shape[1], 1])

        num_superpixel_index = tf.reshape(tf.range(pooled_map_shape[1], dtype=tf.int32),
                                          shape=[1, pooled_map_shape[1], 1])
        num_superpixel_base = tf.ones((pooled_map_shape[0], pooled_map_shape[1], 1), dtype=tf.int32) * num_superpixel_index
        num_superpixel_index = tf.reshape(num_superpixel_base, [pooled_map_shape[0] * pooled_map_shape[1], 1])

        scatter_index = tf.concat([batch_index, num_superpixel_index], axis=-1)

        pooled_feature = tf.scatter_nd(scatter_index, group_feature, shape=pooled_map_shape)
        # ----------------------------- scatter_nd ----------------------------- #

        return pooled_feature


# feature_map = np.random.random((4, 256, 256, 32))
# superpixel_map = np.random.randint(0, 100, (4, 256, 256))
# print(superpixel_map.dtype)
# feature_map = K.variable(value=feature_map)
# superpixel_map = K.variable(value=superpixel_map, dtype=tf.int32)
#
# a = SuperpixelPooling(num_superpixels=100)([feature_map, superpixel_map])
# print(a)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(a))
#
