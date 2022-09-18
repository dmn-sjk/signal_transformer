import tensorflow as tf

from encoder import Encoder


class SignalTransformer(tf.keras.Model):
    def __init__(self, num_signals, num_layers, d_model, num_heads, dff, latent_vector_size, input_signal_length,
                 rate=0.1):
        super(SignalTransformer, self).__init__(name='signal_transformer')

        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='relu'),
            tf.keras.layers.LayerNormalization()
        ])

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_signal_length, input_signal_length, rate)
        self.pooling = tf.keras.layers.AveragePooling1D(d_model, data_format='channels_first')
        self.embedding_gen = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(latent_vector_size)
        ])

        self.call = self.call.get_concrete_function(
            inputs=tf.TensorSpec([None, input_signal_length, num_signals], tf.float32),
            training=tf.TensorSpec([], tf.bool)
        )

    @tf.function
    def call(self, inputs, training):
        projection_output = self.projection(inputs)
        enc_output = self.encoder(projection_output, training, mask=None)  # (batch_size, inp_seq_len, d_model)
        pooling_out = self.pooling(enc_output)
        pooling_out = tf.squeeze(pooling_out, axis=-1)
        embeddings = self.embedding_gen(pooling_out)  # (batch_size, tar_seq_len, target_vocab_size)

        return embeddings

    def warmup(self):
        self(tf.zeros([1, 160, 6], tf.float32), tf.constant(False))


if __name__=="__main__":
    model = SignalTransformer(num_signals=6,
                              num_layers=1,
                              d_model=16,
                              num_heads=2,
                              dff=8,
                              latent_vector_size=256,
                              input_signal_length=160)

    model.warmup()
    model.summary()