from typing import Tuple, Any, Dict, List
import time
import tensorflow as tf
from tensorflow.keras import layers
from model.base import ModelBase
from dataset.base import ImageClassifierDatasetBase


class Gan(ModelBase):
    """Gan model.

    Args:
        dataset (ImageClassifierDatasetBase): dataset object.

    """

    def __init__(
            self,
            dataset: ImageClassifierDatasetBase,
            noise_dims: int,
            generator_optimizer: str = 'adam',
            discriminator_optimizer: str = 'adam',
            generator_lr: float = 1e-4,
            discriminator_lr: float = 1e-4,
            epochs: int = 5,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""

        if int(tf.__version__.split('.')[0]) < 2:
            tf.compat.v1.enable_eager_execution()

        self.dataset = dataset
        self.noise_dims = noise_dims
        self.epochs = epochs

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.generator = self.build_generator(
                            self.noise_dims,
                            self.dataset.input_shape)

        self.discriminator = self.build_discriminator(
                                self.dataset.input_shape)

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
        generator_optimizer = tf.keras.optimizers.get(self.generator_optimizer)
        generator_optimizer._set_hyper("learning_rate", self.generator_lr)
        discriminator_optimizer = tf.keras.optimizers.get(self.discriminator_optimizer)
        discriminator_optimizer._set_hyper("learning_rate", self.discriminator_lr)

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

            generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
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

            avg_gen_loss /= float(self.dataset.steps_per_epoch) / float(self.dataset.batch_size)
            avg_disc_loss /= float(self.dataset.steps_per_epoch) / float(self.dataset.batch_size)
            history['generator_loss'].append(avg_gen_loss.numpy())
            history['discriminator_loss'].append(avg_disc_loss.numpy())

            print('Time for epoch {} is {} sec: generator_loss {}, discriminator_loss {}'.format(
                    epoch + 1,
                    time.time()-start,
                    avg_gen_loss,
                    avg_disc_loss))

        return history

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
