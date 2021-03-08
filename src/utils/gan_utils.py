import tensorflow as tf
from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    Dense,
    Flatten,
    Dropout,
    LeakyReLU,
    Reshape,
    ZeroPadding2D,
    Lambda
)

initializer = 'glorot_uniform'


# ==================================================#
#                   Discriminadores                 #
# ==================================================#

def dcgan_discriminator_stem(input_layer, n_filters):
    stem = Conv2D(
        n_filters,
        (3, 3),
        padding='same',
        kernel_initializer=initializer)(input_layer)
    stem = LeakyReLU(alpha=0.2)(stem)
    return stem


def dcgan_discriminator_block(input_layer, n_filters, n_layers):
    """
        Construye el bloque convolucional para el learner de la GAN 
        Argumentos:
            - input_layer: Capa/Bloque anterior
            - n_filters: Cantidad de feature maps
            - n_layers: Cantidad de capas convolucionales, la primera es con stride 2, las siguienes con stride 1
        
    """
    # Primer bloque
    block = input_layer
    block = Conv2D(
        n_filters,
        (3, 3),
        padding='same',
        strides=(2, 2),
        kernel_initializer=initializer)(block)

    for _ in range(n_layers - 1):
        block = Conv2D(
            n_filters,
            (3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer=initializer)(block)
    block = BatchNormalization(momentum=0.8)(block)
    block = LeakyReLU(alpha=0.2)(block)
    return block


def dcgan_discriminator_learner(input_layer, group_params):
    """
        Construye el learner del discriminador de la GAN
        Argumentos:
            - input_layer: Capa/Bloque anterior
            - group_params: Lista de tuplas con los parámetros de cada bloque.
                            (n_filters,n_layers)
    """
    learner = input_layer
    for n_filters, n_layers in group_params:
        learner = dcgan_discriminator_block(learner, n_filters, n_layers)
    return learner


def dcgan_discriminator_task(input_layer):
    """
        Construye el clasificador para el discriminador de una GAN
    """
    task = input_layer
    task = Flatten()(task)
    task = Dropout(0.4)(task)
    task = Dense(1, activation='sigmoid', kernel_initializer=initializer)(task)
    return task


# ==================================================#
#                   Generadores                     #
# ==================================================#

def dcgan_generator_stem(input_layer, img_dim, n_filters, n_blocks):
    """
        Construye el stem del generador de la GAN
        
        Parámetros:
            - input_layer: Capa/Bloque anterior.
            - img_dim: Dimensiones de la imagen a generar
            - n_filters: Número de canales luego del reshape
            - n_blocks: Número de bloques que tendrá el generador.
                Necesario para definir las neuronas de la primera capa
    """
    # Es necesario considerar el número de bloques que tendrá el generador
    # Como la dimensión de nuestra imagen es de 101x101, hay que considerar
    # la función techo para finalmente llegar a 101.
    initial_dim = img_dim // (2 ** (n_blocks))
    print('initial_dim: {} n_block: {}'.format(initial_dim, n_blocks))
    stem = input_layer
    stem = Dense(n_filters * initial_dim * initial_dim, kernel_initializer=initializer)(stem)
    stem = LeakyReLU(alpha=0.2)(stem)
    stem = Reshape((initial_dim, initial_dim, n_filters))(stem)
    if initial_dim * (2 ** (n_blocks)) < img_dim:
        stem = ZeroPadding2D(((0, 1), (0, 1)))(stem)
    return stem


def dcgan_generator_block(input_layer, n_filters, n_layers):
    """
        Construye el bloque del generador de la GAN.
        Utiliza Conv2DTranspose con Stride 2 para hacer upsampling
        
        Parámetros:
            - input_layer: Capa/Bloque anterior
            - n_filters: Número de filtros
            - n_layers: Número de convoluciones, la primera es Transpose mientras las siguientes
                      son normales 
    """
    block = input_layer
    block = Conv2DTranspose(
        n_filters,
        (4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(block)
    for _ in range(n_layers - 1):
        block = Conv2D(
            n_filters,
            (3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer=initializer)(block)
    block = BatchNormalization(momentum=0.8)(block)
    block = LeakyReLU(alpha=0.2)(block)
    return block


def dcgan_generator_learner(input_layer, g_params):
    """
        Construye el learner del generador de la GAN
        Argumentos:
            - input_layer: Capa/Bloque anterior
            - group_params: Lista con los parámetros de cada bloque
            
    """
    learner = input_layer
    for n_filters, n_layers in g_params:
        learner = dcgan_generator_block(learner, n_filters, n_layers)
    return learner


def dcgan_generator_task(input_layer, channels, img_dim):
    """
        Construye el task del generador de la GAN
    """
    task = input_layer
    task = Conv2DTranspose(channels, (3, 3), activation='tanh', padding='same', kernel_initializer=initializer)(task)
    if task._keras_shape[1] != img_dim:
        task = Lambda(lambda x: tf.image.resize(x, [img_dim, img_dim], method='nearest'))(task)
    return task
