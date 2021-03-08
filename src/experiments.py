import os
import uuid
import errno

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

DATASET_TYPES = ['synthetic', 'real']


def create_folders(path, dataset_name):
    """
        Crea las carpetas donde se almacenará el dataset creado

        Parámetros:
            - path (str): directorio donde se guardará el dataset.
    """
    folder_path = '{}/{}'.format(path, dataset_name)
    images_path = '{}/{}'.format(folder_path, 'images')
    npz_path = '{}/{}'.format(folder_path, 'npz')

    try:
        os.makedirs(folder_path)
        os.makedirs(images_path)
        os.makedirs(npz_path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise

    return folder_path, images_path, npz_path


def save_image(image, images_path, image_id):
    """
        Guarda una imagen en formato png
        
        Parámetros:
            - image (Numpy array): arreglo con la imagen a guardar.
            - images_path (str): path a la carpeta images del dataset.
            - image_id (uuid4): identificador de la imagen.

    """
    image_shape = image.shape
    fig = plt.figure(figsize=(1, 1))
    axes = plt.Axes(fig, [0., 0., 1., 1.])
    axes.set_axis_off()
    fig.add_axes(axes)
    axes.imshow(image[:, :, 0], cmap='hot')
    plt.savefig('{}/{}.png'.format(images_path, image_id), dpi=image_shape[0])
    plt.close()


def create_dataset(
        dataset_name,
        path,
        n_images,
        max_value,
        min_value,
        dataset_type='synthetic',
        g_model=None,
        lens_images=None
):
    if dataset_type not in DATASET_TYPES:
        raise Exception('Tipo de dataset no reconocido, debe ser {}'.format(DATASET_TYPES))
    if dataset_type == DATASET_TYPES[0] and g_model is None:
        raise Exception('Si el dataset es sintético se debe proporcionar un modelo para la generación')
    if dataset_type == DATASET_TYPES[1] and lens_images is None:
        raise Exception(
            'Si el dataset es real se debe proporcionar el conjunto de imágenes desde el cual se construirá')

    if dataset_type == DATASET_TYPES[0]:
        label = 1
    else:
        label = 0
        random_index = np.random.randint(
            0,
            lens_images.shape[0],
            n_images
        )
    labels_list = list()
    try:
        folder_path, images_path, npz_path = create_folders(path, dataset_name)
        for index in tqdm(range(n_images)):
            image_id = uuid.uuid4()
            if label:
                image = g_model.predict(np.random.normal(0, 1, (1, 100)))  # generate image
                image = image * 0.5 + 0.5  # scale between 0 to 1
                save_image(image[0], images_path, image_id)
                image = (max_value - min_value) * image + min_value  # Denormalize image
                np.savez_compressed('{}/{}.npz'.format(npz_path, image_id), image[0, :, :, :])
            else:
                image = lens_images[random_index[index]]
                np.savez_compressed('{}/{}.npz'.format(npz_path, image_id), image)
                image = (image - min_value) / (max_value - min_value)
                save_image(image, images_path, image_id)
            image_label = {'image_id': image_id, 'label': label}
            labels_list.append(image_label)
        # Save labels csv
        df = pd.DataFrame(labels_list)
        df.to_csv('{}/labels.csv'.format(folder_path), index=False, sep=';')
        print('Dataset creado correctamente')
    except Exception as error:
        print('Error al crear el dataset \n {}', format(error))


def read_experiment_dataset(path):
    """
        Leer y retorna un dataset de entrenamiento.

        Parámetros:
            - path (str): directorio del dataset a leer.

        Retorno
            - images_list (Numpy Array): conjunto de imágenes del dataset.
            - df (Pandas DataFrame): dataframe con los identificadores y su respectiva etiqueta.
        """
    images_list = list()
    npz_path = '{}/npz'.format(path)
    df = pd.read_csv('{}/labels.csv'.format(path), sep=';')
    for index, row in df.iterrows():
        npz_id = row['image_id']
        npz = np.load('{}/{}.npz'.format(npz_path, npz_id))['arr_0']
        images_list.append(npz)
    images_list = np.asarray(images_list)
    return images_list, df


def get_activations(images, model):
    """
        Retorna las activaciones de la red Lensfinder de un conjunto de imágenes

        Parámetros:
            - images (Numpy array): conjunto de imágenes.
            - model (Keras Model): Red lensfinder
        
        Retorno:
            - images_activation_list (list): lista con las activaciones
    """
    activations = model.predict(images)
    images_activation_list = list()
    for index in range(images.shape[0]):
        images_activation_list.append((images[index], activations[index]))
    return images_activation_list


def get_nearest_images(synthetic_images, training_images, model, distance_function):
    """
        Calcula la imagen más cercana del conjunto de entrenamiento para 
        cada imagen generada

        Parámetros:
            - synthetic_images (Numpy array): imágenes generadas.
            - training_images (Numpy array): conjunto de imágenes de entrenamiento.
            - model (Keras Model): red Lensfinder.
            - distance_function (sklearn metric function): función para calcular la distancia entre vectores
        
        Retorno:
            - pairs_of_images (list): lista con los pares de imágenes más cercanas    
    """
    pairs_of_images = list()
    synthetic_activation_list = get_activations(synthetic_images, model)
    training_activation_list = get_activations(training_images, model)

    for synthetic_image, synthetic_activation in synthetic_activation_list:
        min_distance = 1000000000000
        nearest_image = None
        for training_image, training_activation in training_activation_list:
            actual_distance = distance_function(synthetic_activation.reshape(1, -1), training_activation.reshape(1, -1))
            if actual_distance < min_distance:
                min_distance = actual_distance
                nearest_image = training_image
        pairs_of_images.append((synthetic_image, nearest_image))
    return pairs_of_images


def plot_pairs_of_images(pairs_list, image_path):
    """
        Grafica y guarda las imágenes más cercanas.

        Parámetros:
        - pairs_list (list): lista de pares más cercanos.
        - image_path (str): directorio donde se guardará la imagen.
    """
    fig, axes = plt.subplots(len(pairs_list), 2, figsize=(5, 80))
    for index in range(len(pairs_list)):
        axes[index][0].imshow(pairs_list[index][0][:, :, 0], cmap='hot')
        axes[index][1].imshow(pairs_list[index][1][:, :, 0], cmap='hot')
        axes[index][0].axis('off')
        axes[index][1].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(image_path, bbox_inches='tight')
    plt.show()
