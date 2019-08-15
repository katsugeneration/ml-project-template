from typing import Tuple, Any, Dict, List, Union
import time
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from model.base import ModelBase
from dataset.base import ImageClassifierDatasetBase


class Gan(ModelBase):
    """Gan model.

    Args:
        dataset (ImageClassifierDatasetBase): dataset object.
        noise_dims (int): noise vector dimmension size.
        epochs (int): numaber of training epochs.
        generator_optimizer_name (str): generator otimizer class name.
        discriminator_optimizer_name (str): discriminator otimizer class name.
        generator_lr (float): generator learning rate.
        discriminator_lr (float): discriminator learning rate.

    """

    def __init__(
            self,
            dataset: ImageClassifierDatasetBase,
            noise_dims: int,
            epochs: int = 5,
            generator_optimizer_name: str = 'adam',
            discriminator_optimizer_name: str = 'adam',
            generator_lr: float = 1e-4,
            discriminator_lr: float = 1e-4,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""

        if int(tf.__version__.split('.')[0]) < 2:
            tf.compat.v1.enable_eager_execution()

        self.dataset = dataset
        self.noise_dims = noise_dims
        self.epochs = epochs

        self.generator_optimizer_name = generator_optimizer_name
        self.discriminator_optimizer_name = discriminator_optimizer_name
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.generator = self.build_generator(
                            self.noise_dims,
                            self.dataset.input_shape)

        self.discriminator = self.build_discriminator(
                                self.dataset.input_shape)

        self.compile(self.generator, self.discriminator)

    def compile(
            self,
            generator: tf.keras.Model,
            discriminator: tf.keras.Model) -> None:
        """Set optimizer to model.

        Args:
            generator (tf.keras.Model): generator model object.
            discriminator (tf.keras.Model): discriminator model object.
        """
        self.gen_optimizer = tf.keras.optimizers.get(self.generator_optimizer_name)
        self.gen_optimizer._set_hyper("learning_rate", self.generator_lr)
        self.disc_optimizer = tf.keras.optimizers.get(self.discriminator_optimizer_name)
        self.disc_optimizer._set_hyper("learning_rate", self.discriminator_lr)

        self.checkpoint = tf.train.Checkpoint(
                                generator_optimizer=self.gen_optimizer,
                                discriminator_optimizer=self.disc_optimizer,
                                generator=generator,
                                discriminator=discriminator)

    def train(self) -> Dict[str, List[Any]]:
        """Training model.

        Return:
            log (Dict[str, List[Any]]): training log.

        """
        history: Dict[str, List[Any]] = {
            'generator_loss': [],
            'discriminator_loss': []
        }

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        def discriminator_loss(real_output, fake_output):
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss

        def generator_loss(fake_output):
            return cross_entropy(tf.ones_like(fake_output), fake_output)

        @tf.function
        def train_step(
                images: List) -> Tuple[int, int]:
            noise = tf.random.normal([self.dataset.batch_size, self.noise_dims])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)

                real_output = self.discriminator(images, training=True)
                fake_output = self.discriminator(generated_images, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
            self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
            return gen_loss, disc_loss

        data_generator = self.dataset.training_data_generator()
        for epoch in range(self.epochs):
            start = time.time()
            avg_gen_loss = tf.constant(0.)
            avg_disc_loss = tf.constant(0.)
            i = 0
            while i <= self.dataset.steps_per_epoch:
                image_batch = next(data_generator)
                gen_loss, disc_loss = train_step(image_batch[0])
                avg_gen_loss += gen_loss
                avg_disc_loss += disc_loss
                i += 1

            avg_gen_loss /= float(self.dataset.steps_per_epoch) * float(self.dataset.batch_size)
            avg_disc_loss /= float(self.dataset.steps_per_epoch) * float(self.dataset.batch_size)
            history['generator_loss'].append(avg_gen_loss.numpy())
            history['discriminator_loss'].append(avg_disc_loss.numpy())

            print('Time for epoch {} is {} sec: generator_loss {}, discriminator_loss {}'.format(
                    epoch + 1,
                    time.time()-start,
                    avg_gen_loss,
                    avg_disc_loss))

        return history

    def predict(self) -> Tuple[List[np.array], List[np.array]]:
        """Predict model.

        Return:
            predicts (List[np.array]): generated images.
            gt (List[np.array]): ground truth images.

        """
        noise = tf.random.normal([self.dataset.batch_size, self.noise_dims])
        generated_images = self.generator(noise, training=False)
        generated_images = generated_images.numpy() * 127.5 + 127.5
        x_test, _ = self.dataset.eval_data()
        x_test = x_test[np.random.choice(x_test.shape[0], self.dataset.batch_size, replace=False)]
        x_test = x_test * 127.5 + 127.5
        return generated_images, x_test

    def save(
            self,
            path: Union[str, pathlib.Path]) -> None:
        """Save model.

        Args:
            path (str or pathlib.Path): path to model save directory.

        """
        self.checkpoint.save(file_prefix=str(path))

    def build_generator(
            self,
            noise_dims: int,
            input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        H = input_shape[0] // 4
        W = input_shape[1] // 4
        LAST_C = input_shape[2]
        channels = [256, 128, 64]

        inputs = tf.keras.layers.Input(shape=(noise_dims, ))
        x = layers.Dense(
                H * W * channels[0],
                use_bias=False,
                kernel_initializer='he_normal')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((H, W, 256))(x)

        x = layers.Conv2DTranspose(
                channels[1],
                (5, 5),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(
                channels[2],
                (5, 5),
                strides=(2, 2),
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(
                LAST_C,
                (5, 5),
                strides=(2, 2),
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                activation='tanh')(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def build_discriminator(
            self,
            input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)

        channels = [64, 128]
        x = layers.Conv2D(
                channels[0],
                (5, 5),
                strides=(2, 2),
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal')(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(
                channels[1],
                (5, 5),
                strides=(2, 2),
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(
                    1,
                    use_bias=False,
                    kernel_initializer='he_normal')(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model
