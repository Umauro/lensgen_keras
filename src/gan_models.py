import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

from utils.gan_utils import (
    dcgan_discriminator_stem,
    dcgan_discriminator_learner,
    dcgan_discriminator_task,
    dcgan_generator_stem,
    dcgan_generator_learner,
    dcgan_generator_task
)


def label_noise(labels, frac):
    """
        Invierte una cierta fracción de las etiquetas

        Parámetros:
            - labels: Etiquetas del batch.
            - frac: Fracción de las etiquetas a invertir.
    """
    y_shape = labels.shape[0]
    # Fraction of labels to change
    n_labels = int(frac * y_shape)
    # Create Mask
    selected_labels = np.random.choice([i for i in range(y_shape)], size=n_labels)
    # Flip labels
    labels[selected_labels] = 1 - labels[selected_labels]
    return labels


def print_progress(epoch, loss_synthetic, loss_real, loss, g_loss):
    """
        Muestra un mini reporte las loss y accuracy en 1 epoch

        Atributos
            - Epoch: Epoch actual
            - loss_synthetic: Loss del discriminador en el batch sintético.
            - loss_real: Loss del discriminador en el batch real.
            - loss: Loss del discriminador
            - g_loss: Loss del generador
    """
    print(
        """
            =====================================================================================
                Loss Synthetic: {} - Loss Real: {} \n
                Acc Synthetic: {} - Acc Real: {}
                         Generator Acc: {}
            =====================================================================================
        """.format(
            np.round(loss_synthetic[0], 4),
            np.round(loss_real[0], 4),
            np.round(loss_synthetic[1], 4),
            np.round(loss_real[1], 4),
            np.round(g_loss[1], 4)
        )
    )
    print(
        """
        Epoch: {} - Discriminator loss:{} - Discriminator Accuracy: {} - Generator loss:{}
        """.format(epoch + 1, loss[0], loss[1], g_loss[0])
    )


class GanUtils:
    """
        Clase con métodos útiles para todas las GAN implementadas
    """

    def __init__(self, latent_dim):
        self.d_loss = list()
        self.g_loss = list()
        self.d_acc = list()
        self.g_acc = list()
        self.x_max = 0
        self.x_min = 0
        self.latent_dim = latent_dim
        self.noise_for_previews = np.random.normal(0, 1, (9, self.latent_dim))

    def preprocess_images(self, images, clip=False):
        """
            Pre-procesamiento de las imágenes, utilizando la media y la desviación estándar del conjunto
            de entrenamiento.
            Como la DCGAN utiliza Tangente Hiperbólica como activación es necesario contar con imágenes con valores
            entre [-1,1]
            
            Atributos: 
                - images: Conjunto de imágenes
                - dataset: Dataset a utilizar
            Retorno:
                - Conjunto de imágenes normalizadas entre -1 y 1.
        """
        if self.dataset == 'lens':
            images[images == 100] = 0  # fix masked pixels
            if clip:
                images = np.clip(images, -1e-11, 1e-9)
            self.x_max = np.max(images)
            self.x_min = np.min(images)
            images = (images - self.x_min) / (self.x_max - self.x_min)  # scale between 0,1
            return images * 2 - 1  # re-scaled between -1,1
        if self.dataset == 'Cifar10':
            return (images - 127.5) / 127.5
        else:
            raise Exception('Dataset sin preprocesamiento implementado')

    def save_images(self, path):
        """
            Guarda en disco una grilla con 100 imágenes generadas
        """
        images = self.g_model.predict(np.random.normal(0, 1, (100, self.latent_dim)))
        images = images * 0.5 + 0.5  # [-1,1] to [0,1]
        plt.figure(figsize=(30, 30))  # Define figure size
        for index in range(100):
            plt.subplot(10, 10, index + 1)
            plt.imshow(images[index, :, :, 0], cmap='hot')
        plt.savefig(path)

    def plot_previews(self, images, epoch_index):
        """
            Muestra 9 imágenes sintéticas y 9 imágenes reales
            a partir de un vector de ruido fijo
        """

        # Synthetic Images
        synthetic_images = self.g_model.predict(
            self.noise_for_previews
        )

        # Real Images
        random_index = np.random.randint(0, images.shape[0], 9)
        random_images = images[random_index]

        fig, axes = plt.subplots(2, 9, figsize=(30, 7))
        fig.suptitle('Synthetic images - Epoch {}'.format(epoch_index))
        for img in range(9):
            if self.input_shape[2] == 1:
                axes[0][img].imshow(synthetic_images[img, :, :, 0] * 0.5 + 0.5, cmap='hot')
                axes[1][img].imshow(random_images[img, :, :, 0] * 0.5 + 0.5, cmap='hot')
            elif self.input_shape[2] == 3:
                axes[0][img].imshow(synthetic_images[img, :, :, :] * 0.5 + 0.5, cmap='hot')
                axes[1][img].imshow(random_images[img, :, :, :] * 0.5 + 0.5, cmap='hot')
            else:
                raise Exception('Número de canales no soportado')
            axes[0][img].set_title('Synthetic')
            axes[1][img].set_title('Real')
        plt.show()

    def plot_training_process(self):
        """
            Muestra el progreso del entrenamiento
        """
        epochs = [epoch + 1 for epoch in range(len(self.d_loss))]
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        # Losses
        axes[0].plot(epochs, self.d_loss, '.-')
        axes[0].plot(epochs, self.g_loss, '.-')
        axes[0].legend(['Discrimnator Loss', 'Generator Loss'])
        axes[0].set_xlabel('Train step')
        axes[0].set_ylabel('Loss')
        axes[0].set_ylim([0, 7])  # Y range between 0 and 7
        # Acc
        axes[1].plot(epochs, self.d_acc, '.-')
        axes[1].plot(epochs, self.g_acc, '.-')
        axes[1].legend(['Discrimnator Acc', 'Generator Acc'])
        axes[1].set_xlabel('Train step')
        axes[1].set_ylabel('Acc')
        plt.show()

    def plot_prediction_on_synthetic_data(self, g_noise):
        fig, axes = plt.subplots(1, 1, figsize=(15, 7))
        axes.hist(
            np.asarray(
                self.gan_model.predict(g_noise)
            ).flatten(),
            range=(0, 1)
        )
        axes.legend(['Discriminator prediction for Synthetic images'])
        plt.show()


class LensGenV1(GanUtils):
    """
        LensGEN para generación de imágenes sintéticas.
        Estructura basada desde el curso Keras Idiomatic Programmer de GoogleCloud
        https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer
    """

    def __init__(
            self,
            latent_dim,
            input_shape,
            g_initial_filters,
            d_initial_filters,
            d_params,
            g_params,
            dataset,
            lr):
        """
            Construye el LensGEN
            Atributos:
                - latent_dim: Dimensión del espacio latente
                - input_shape: Dimensiones de la imagen a generar.
                - g_initial_filters: Cantidad de filtros para la primera capa del generador.
                - d_initial_filters: Cantidad de filtros para la primera capa del discriminador. 
                - d_optimizer: Optimizador para el entrenamiento del discriminador.
                - g_optimizer: Optimizador para el entrenamiento del generador.
                - d_params: Parámetros para el discriminador.
                - g_params: Parámetros para el discriminador
        """
        super().__init__(latent_dim)
        self.input_shape = input_shape
        self.d_params = d_params
        self.g_params = g_params
        self.g_initial_filters = g_initial_filters
        self.d_initial_filters = d_initial_filters
        self.dataset = dataset
        self.lr = lr

        # Build Discriminator Architecture
        self.d_model = self.build_discriminator()
        # Build Generator Architecture
        self.g_model = self.build_generator()

        # Build GAN
        self.gan_model = self.build_gan()
        self.gan_model.summary()

    def build_discriminator(self):
        """
            Construye el discriminador de la GAN
            Basado en la DCGAN
        """
        input_layer = Input(shape=self.input_shape)
        discriminator = dcgan_discriminator_stem(input_layer, self.d_initial_filters)
        discriminator = dcgan_discriminator_learner(discriminator, self.d_params)
        discriminator = dcgan_discriminator_task(discriminator)
        d_model = Model(input_layer, discriminator)
        d_model.compile(
            loss=BinaryCrossentropy(label_smoothing=0.1),
            optimizer=Adam(self.lr, 0.5),
            metrics=['accuracy']
        )

        return d_model

    def build_generator(self):
        """
            Construye el generador de la GAN
            Basado en la DCGAN
        """
        input_layer = Input((self.latent_dim,))  # El generador recibe el ruido
        generator = dcgan_generator_stem(
            input_layer,
            self.input_shape[0],
            self.g_initial_filters,
            len(self.g_params)
        )
        generator = dcgan_generator_learner(generator, self.g_params)
        generator = dcgan_generator_task(
            generator,
            channels=self.input_shape[2],
            img_dim=self.input_shape[0]
        )
        g_model = Model(input_layer, generator)
        return g_model

    def build_gan(self):
        """
            Construye la GAN. Compuesta por generador y discriminador
        """
        self.d_model.trainable = False

        noise_input = Input((self.latent_dim,))

        # Capa que entrega las imágenes sintéticas
        gen_img = self.g_model(noise_input)

        # Discriminador recibe las imágenes
        discriminator = self.d_model(gen_img)

        gan_model = Model(noise_input, discriminator)
        gan_model.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(self.lr, 0.5),
            metrics=['accuracy']
        )
        return gan_model

    def train_step(self, d_steps, batch_size, index, images, tqdm_bar):
        """
            Ejecuta un paso de entrenamiento de la GAN

            Parametros:
                - epoch (int): Epoch de entrenamiento.
                - d_steps (int): Número de ajustes del discriminador por cada paso.
                - batch_size (int): Tamaño del mini-batch a utilizar
                - index (int): Paso de entrenamiento 
                - images (numpy array): Conjunto de imágenes de entrenamiento
                - tqdm_bar (tqdm object): Barra de progreso.  
        """
        for step in range(d_steps):
            first_index = index * batch_size
            last_index = (index + 1) * batch_size
            # Imágenes reales para entrenamiento
            if last_index < len(images):
                batch = images[first_index:last_index]
            else:
                # No pasarse
                batch = images[first_index:len(images) - 1]
            # NoisyLabels
            synthetic_labels = label_noise(np.zeros((len(batch), 1)), 0.05)
            real_labels = label_noise(np.ones((len(batch), 1)), 0.05)

            # Sampleo desde el espacio latente
            g_noise = np.random.normal(0, 1, (len(batch), self.latent_dim))
            synthetic_images = self.g_model.predict(g_noise)

            # ==================================#
            # Entrenamiento del discriminador  #
            # ==================================#
            loss_real = self.d_model.train_on_batch(
                batch,
                real_labels
            )

            loss_synthetic = self.d_model.train_on_batch(
                synthetic_images,
                synthetic_labels
            )

            loss = 0.5 * np.add(loss_real, loss_synthetic)

            if step == d_steps - 1:
                self.d_loss.append(loss[0])
                self.d_acc.append(loss[1])
        # =================================#
        # Entrenamiento del generador     #
        # =================================#

        g_noise = np.random.normal(0, 1, (len(batch), self.latent_dim))
        g_loss = self.gan_model.train_on_batch(g_noise, real_labels)
        self.g_loss.append(g_loss[0])
        self.g_acc.append(g_loss[1])

        tqdm_bar.set_postfix(d_loss=loss[0], acc=loss[1], g_loss=g_loss[0])
        return loss_synthetic, loss_real, loss, g_loss, g_noise

    def train(self, images, batch_size, epochs, d_steps=1, plot_images=False):
        """
            Entrenamiento de la GAN, siguiendo el algoritmo de Goodfellow
            Atributos: 
                - images: Imágenes de entrenamiento.
                - batch_size: Tamaño de los batch a utilizar
                - epochs: Cantidad de epochs
                - d_steps: Cantidad de pasos de entrenamiento del discriminador por cada epoch
        """
        for epoch in range(epochs):
            step_len = len(images) // batch_size
            # Si el tamaño del step len no alcanza al dataset completo, hacemos un índice más
            if step_len * batch_size < images.shape[0]:
                step_len += 1
            tqdm_bar = tqdm(range(step_len))
            for index in tqdm_bar:
                tqdm_bar.set_description('Epoch: {}'.format(epoch + 1))
                loss_synthetic, loss_real, loss, g_loss, g_noise = self.train_step(
                    d_steps,
                    batch_size,
                    index,
                    images,
                    tqdm_bar
                )

            print_progress(epoch, loss_synthetic, loss_real, loss, g_loss)

            if plot_images and ((epoch + 1) % 5 == 0):
                if epoch != 0:
                    self.plot_training_process()
                    self.plot_prediction_on_synthetic_data(g_noise)
                self.plot_previews(images, epoch)
