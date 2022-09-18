import tensorflow as tf
import pickle


def tf_dataset(dataset, start_cut=0, cut_length=160):
    def generator():
        for step in dataset:
            s = step['signal'][start_cut:(start_cut + cut_length)]
            signal = tf.convert_to_tensor(s, tf.float32)
            position = tf.convert_to_tensor(step['position_optitrack'], tf.float32)
            yield signal, position

    return tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([None, 6]), tf.TensorShape([3]))
    )


def load_dataset(path, batch_size):
    with open(path, 'rb') as fp:
        dataset = pickle.load(fp)

    # load data
    train_ds = tf_dataset(dataset['train_ds'], 0, 160) \
        .shuffle(512) \
        .padded_batch(batch_size, ([None, 6], [3]))
    val_ds = tf_dataset(dataset['val_ds'], 0, 160) \
        .shuffle(512) \
        .padded_batch(batch_size, ([None, 6], [3]))

    return train_ds, val_ds
