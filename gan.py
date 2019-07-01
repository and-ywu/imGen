import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import summary
import datetime
import os
import h5py

#read in h5 files
f = h5py.File('data/smolfilterednormalized.h5', 'r')
data = f['images']

class GAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 200  # dimension of the noise

        optimizer = Adam(0.0001, 0.5)
        dis_optimizer = Adam(0.0001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=dis_optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        model._name="generator3"
        model.add(Dense(256 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 256)))
        model.add(UpSampling2D())
        model.add(Conv2D(512, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        #model.add(Activation("relu"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        #model.add(Activation("relu"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        #model.add(Activation("relu"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        #model.add(Activation("relu"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(self.channels, kernel_size=5, padding="same", activation = 'tanh'))

        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)
        
    def build_discriminator(self):
        dr = 0.1	# dropout rate
        model = Sequential()
        model._name="discriminator2"
        model.add(Conv2D(64, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(dr))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        #model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(dr))
        model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(dr))
        model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dr))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=16, sample_interval=50):
        current_time = str(datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"))
        gen_log_dir = 'logs/tensorboard/train/gen/' + current_time
        dis_real_log_dir = 'logs/tensorboard/train/disreal/' + current_time
        dis_fake_log_dir = 'logs/tensorboard/train/disfake/' + current_time
        self.gen_summary = summary.create_file_writer(gen_log_dir)
        self.dis_real_summary = summary.create_file_writer(dis_real_log_dir)
        self.dis_fake_summary = summary.create_file_writer(dis_fake_log_dir)
        
        # Load the dataset
        X_train = data
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))
        noise_prop=0.05

        for epoch in range(epochs):

            # Select a random batch of images
            for i in range(5):
                #idx = np.random.randint(0, X_train.shape[0], batch_size)
                #imgs = X_train[idx]
                imgs = X_train[[0]]

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                #train Discriminator
                #if epoch % sample_interval == 0:
                #    print(self.discriminator.predict(np.array([imgs[0]])),)
                #    print(self.discriminator.predict(np.array([gen_imgs[0]])))
                true_labels = zeros + np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
                #flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop*len(true_labels)))
                #true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
                gen_labels = ones - np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
                #flipped_idx = np.random.choice(np.arange(len(gen_labels)), size=int(noise_prop*len(gen_labels)))
                #gen_labels[flipped_idx] = 1 - gen_labels[flipped_idx]
                d_loss_real = self.discriminator.train_on_batch(imgs, true_labels)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, gen_labels)

            #  Train Generator
            for i in range(1):
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, zeros)
    
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.log(epoch, d_loss_real, d_loss_fake, g_loss)
        self.log(epoch+1, d_loss_real, d_loss_fake, g_loss)
            
    def log(self, epoch, d_loss_real, d_loss_fake, g_loss):
        #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        print ("%d [D real: %f, acc: %.2f%%] [D fake: %f, acc: %.2f%%] [G: %f]" 
               % (epoch, d_loss_real[0], 100*d_loss_real[1], d_loss_fake[0], 100*d_loss_fake[1], g_loss))
        with self.gen_summary.as_default():
            summary.scalar('loss', g_loss, step=epoch)
        with self.dis_real_summary.as_default():
            summary.scalar('loss', d_loss_real[0], step=epoch)
            summary.scalar('accuracy', d_loss_real[1], step=epoch)
        with self.dis_fake_summary.as_default():
            summary.scalar('loss', d_loss_fake[0], step=epoch)
            summary.scalar('accuracy', d_loss_fake[1], step=epoch)
        self.sample_images2(epoch)

    def sample_images(self, epoch):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("gan/%d.png" % epoch)
        plt.close()

    def sample_images2(self, epoch):
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        real_imgs = np.array([np.expand_dims(data, axis=3)[np.random.randint(0, data.shape[0])]])
        #print(np.min(gen_imgs), np.max(gen_imgs))
        #print(self.discriminator.predict(real_imgs), self.discriminator.predict(gen_imgs))
        fig = plt.subplots(figsize=(3,3))
        plt.imshow(np.squeeze(gen_imgs), cmap='Greys')
        plt.show()
        plt.close()

epochs = 10000
sample_interval = 100
gan = GAN()
gan.train(epochs=epochs, batch_size=1, sample_interval=sample_interval)