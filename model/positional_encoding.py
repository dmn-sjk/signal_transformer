import numpy as np
import tensorflow as tf


class PositionalEmbedding(tf.keras.Model):
    def __init__(self, dict_size, max_sequence_size, embedding_size):
        super(PositionalEmbedding, self).__init__(name='positional_embedding')

        self.embedding_size = embedding_size
        self.embedding = tf.keras.layers.Embedding(dict_size, embedding_size)
        self.embedding_positional = positional_encoding(max_sequence_size, embedding_size)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        in_shape = tf.shape(inputs)

        positions = tf.range(0, in_shape[1])[tf.newaxis]
        positions = tf.tile(positions, (in_shape[0], 1))

        dict_emb = self.embedding(positions)
        dict_emb *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        pos_emb = tf.gather(self.embedding_positional, positions)

        return dict_emb + inputs

    def warmup(self):
        with tf.name_scope(self.name):
            self.embedding.build(None)
            self.embedding_positional.build(None)
            self.built = True


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads, dtype=tf.float32)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates
