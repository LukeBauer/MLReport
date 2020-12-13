import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import random_normal
from tensorflow.keras import Model

from matplotlib import pyplot as plt


class Samp(Layer):
    def call(self, inputs):
        m =inputs[0]
        v=inputs[1]
        epsilon = random_normal(shape=(tf.shape(m)[0], tf.shape(m)[1]))
        return m + tf.exp(0.5 * v) * epsilon


def initenc():

    encin = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3,3), activation="relu", strides=(2,2), padding="same")(encin)
    x = Conv2D(64, (3,3), activation="relu", strides=(2,2), padding="same")(x)

    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)

    mean = Dense(2)(x)
    var = Dense(2)(x)

    s = Samp()([mean, var])
    encoder = Model(encin, [mean, var, s], name="encoder")
    return encoder

def initdec():
    decin = Input(shape=(2,))
    x = Dense(7 * 7 * 32, activation="relu")(decin)
    x = Reshape((7, 7, 32))(x)

    x = Conv2DTranspose(64, (3,3), activation="relu", strides=2, padding="same")(x)
    x = Conv2DTranspose(32, (3,3), activation="relu", strides=2, padding="same")(x)

    decout = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = Model(decin, decout, name="decoder")
    return decoder


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            mean, var, z = self.encoder(data)
            output = self.decoder(z)
            output_loss = tf.reduce_mean(
                binary_crossentropy(data, output)
            )
            output_loss *= 28 * 28
            kl_loss = 1 + var - tf.square(mean) - tf.exp(var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = output_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": output_loss,
            "kl_loss": kl_loss,
        }

def plot(encoder, decoder):

    n = 30
    digit_size = 28
    scale = 2.0
    figsize = 15
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
    plt.savefig("vaesamp.png")
    
def train():

    (x_train, _), (x_test, _) =mnist.load_data()
    data = np.concatenate([x_train, x_test], axis=0)
    data = np.expand_dims(data, -1).astype("float32") / 255

    encoder=initenc()
    decoder=initdec()
    vae = VAE(encoder, decoder)

    vae.compile(optimizer=Adam(lr=0.0002, beta_1=0.5))
    vae.fit(data, epochs=30, batch_size=128)

    plot(encoder, decoder)
if __name__ == '__main__':
    train()

