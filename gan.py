from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot

def load_mnist():
	(trainX, _), (_, _) = mnist.load_data()
	X = expand_dims(trainX, axis=-1)
	X = X.astype('float32')
	X = X / 255.0
	return X
 
def generator():
	model = Sequential()
	#nodes = 128 * 7 * 7
	model.add(Dense(128*7*7, input_dim=100))
	model.add(BatchNormalization())
	model.add(ReLU())
	model.add(Reshape((7, 7, 128)))

	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(BatchNormalization())
	model.add(ReLU())

	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(BatchNormalization())
	model.add(ReLU())



	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model
 

def discriminator(in_shape=(28,28,1)):
	model = Sequential()

	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(.4))

	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=.2))
	model.add(Dropout(.4))

	# model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	# model.add(LeakyReLU(alpha=.2))
	# model.add(Dropout(.4))

	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
 

def gan(gen, disc):
	disc.trainable = False
	model = Sequential()
	model.add(gen)
	model.add(disc)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
 
 
def gen_fake(gen, n_samples):
	x = randn(100 * n_samples)
	x  = x.reshape(n_samples, 100)
	x =gen.predict(x)
	return x
 
def save_plot(examples, epoch, n=10):
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()
 
def eval(epoch, gen, disc, dataset, n_samples=100):
	
	x_real = dataset[randint(0, dataset.shape[0], n_samples)]
	y_real = ones((n_samples, 1))

	x_fake = gen_fake(gen, n_samples)
	y_fake = zeros((n_samples, 1))

	_, acc_real = disc.evaluate(x_real, y_real, verbose=0)
	_, acc_fake = disc.evaluate(x_fake, y_fake, verbose=0)

	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	save_plot(x_fake, epoch)
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	gen.save(filename)
 
def train():
	disc = discriminator()
	gen = generator()
	ganm = gan(gen, disc)
	dataset = load_mnist()
	
	n_epochs=100
	batch=256

	run = int(dataset.shape[0] / batch)
	half_batch = int(batch / 2)
	for i in range(n_epochs):
		for j in range(run):
			x_real = dataset[randint(0, dataset.shape[0], half_batch)]
			y_real = ones((half_batch, 1))

			x_fake = gen_fake(gen, half_batch)
			y_fake = zeros((half_batch, 1))

			x, y = vstack((x_real, x_fake)), vstack((y_real, y_fake))
			d_loss, _ = disc.train_on_batch(x, y)

			x_gen = randn(100 *batch)
			x_gen  = x_gen.reshape(batch, 100)
			y_gen = ones((batch, 1))

			gan_loss = ganm.train_on_batch(x_gen, y_gen)
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, run, d_loss, gan_loss))

		eval(i, gen, disc, dataset)
 
if __name__ == '__main__':
	train()
