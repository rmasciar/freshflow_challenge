import tensorflow as tf
import numpy as np

class Trainer:
    def __init__(self):
        self.model = None
        self.history = None

    def windowed_dataset(self, series, window_size, batch_size, shuffle_buffer):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size + 1))
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda w: (w[:-1], w[1:]))
        return ds.batch(batch_size).prefetch(1)

    def train(self, train_x, window_size:int=10, batch_size:int=32, shuffle_buffer:int=1000, n_epochs:int=100):
        tf.keras.backend.clear_session()
        tf.random.set_seed(51)
        np.random.seed(51)

        dataset = self.windowed_dataset(train_x, window_size, batch_size, shuffle_buffer)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                                   strides=1, padding="causal",
                                   activation="relu",
                                   input_shape=[None, 3]),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dense(30, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(3),
        ])

        self.model.compile(loss=tf.keras.losses.Huber(),
                           optimizer="Adam",
                           metrics=["mae"])
        self.history = self.model.fit(dataset, epochs=n_epochs)