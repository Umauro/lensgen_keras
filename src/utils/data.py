import time
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits


def timer_decorator(function):
    """
        Decorador para calcular el tiempo de ejecución
        de una función
    """

    def override_function(*args, **kw):
        start_time = time.time()
        result = function(*args, **kw)
        elapsed = time.time() - start_time
        print('Tiempo de ejecución: {} segundos'.format(elapsed))
        return result

    return override_function


@timer_decorator
def read_ground_based_images(path, df):
    """
        Leer las imágenes de la categoría Ground Based

        Parámetros:
            - path (str): Path a la carpeta con el dataset
            - df (DataFrame): DataFrame con las etiquetas del dataset
    """
    images = list()  # number of images, dim_1, dim_2, channels
    bands = ['R', 'I', 'G', 'U']
    for index, row in df.iterrows():
        image_id = int(row['ID'])
        image_array = np.zeros((101, 101, 4))
        for band_number, band in enumerate(bands):
            image_file = get_pkg_data_filename(
                '{}Band{}/imageSDSS_{}-{}.fits'.format(
                    path,
                    band_number + 1,
                    band,
                    image_id
                )
            )
            image_data = fits.getdata(image_file, ext=0)
            image_array[:, :, band_number] = image_data
        images.append(image_array)
    images = np.asarray(images)
    return images


@timer_decorator
def read_space_based_images(path, df):
    """
        Leer las imágenes de la categoría Space Based

        Parámetros:
            - path (str): Path a la carpeta con el dataset
            - df (DataFrame): DataFrame con las etiquetas del dataset
    """
    images = list()
    for index, row in df.iterrows():
        image_id = int(row['ID'])
        image_file = get_pkg_data_filename(
            '{}imageEUC_VIS-{}.fits'.format(
                path,
                image_id
            )
        )
        image_data = fits.getdata(image_file, ext=0)
        images.append(image_data)
    images = np.asarray(images)
    images = np.reshape(images, (len(df), 101, 101, 1))  # reshape to consider 1 channel
    return images


@timer_decorator
def read_space_lens_images(path, df, lens_label):
    """
        Leer las imágenes con o sin evidencia de lentes de la categoría Space Based

        Parámetros:
            - path (str): Path a la carpeta con el dataset
            - df (DataFrame): DataFrame con las etiquetas del dataset
            - lens_label (int): 1 para leer imagenes con lentes, 0 sin evidencia. 
    """
    images = list()
    lens_counter = 0
    for index, row in df.iterrows():
        if row['is_lens'] == lens_label:
            lens_counter += 1
            image_id = int(row['ID'])
            image_file = get_pkg_data_filename(
                '{}imageEUC_VIS-{}.fits'.format(
                    path,
                    image_id
                )
            )
            image_data = fits.getdata(image_file, ext=0)
            images.append(image_data)
    images = np.asarray(images)
    images = np.reshape(images, (lens_counter, 101, 101, 1))
    return images


@timer_decorator
def read_space_based_in_chunks(path, df, chunksize=50000):
    chunks_list = list()
    for _, chunk in df.groupby(np.arange(len(df)) // chunksize):
        images = list()
        for index, row in chunk.iterrows():
            image_id = int(row['ID'])
            image_file = get_pkg_data_filename(
                '{}imageEUC_VIS-{}.fits'.format(
                    path,
                    image_id
                )
            )
            image_data = fits.getdata(image_file, ext=0)
            images.append(image_data)
        images = np.asarray(images)
        images = np.reshape(images, (len(chunk), 101, 101, 1))
        chunks_list.append(images)
    return chunks_list


@timer_decorator
def read_space_based_lens_in_chunks(path, df, chunk_size=50000):
    chunks_list = list()
    for _, chunk in df.groupby(np.arange(len(df)) // chunk_size):
        images = list()  # number of images, dim_1, dim_2, channels
        counter = 0
        for index, row in chunk.iterrows():
            if row['no_source'] == 0:
                counter += 1
                image_id = int(row['ID'])
                image_file = get_pkg_data_filename(
                    '{}imageEUC_VIS-{}.fits'.format(
                        path,
                        image_id
                    )
                )
                image_data = fits.getdata(image_file, ext=0)
                images.append(image_data)
        images = np.asarray(images)
        images = np.reshape(images, (counter, 101, 101, 1))
        chunks_list.append(images)
    return chunks_list


def preprocess_image(images, mean, std):
    images[images == 100] = 0.0
    images = (images - mean) / std
    return images


def proba_to_label(probas, threshold=0.5):
    labels = np.copy(probas)
    mask = (probas >= threshold)
    labels[mask] = 1
    labels[~mask] = 0
    return labels


@timer_decorator
def get_test_predictions(path, df, model, train_mean, train_std, chunksize=10000):
    y = list()
    y_pred = list()
    y_pred_classes = list()
    for _, chunk in df.groupby(np.arange(len(df)) // chunksize):
        images = list()
        for index, row in chunk.iterrows():
            image_id = int(row['ID'])
            image_file = get_pkg_data_filename(
                '{}imageEUC_VIS-{}.fits'.format(
                    path,
                    image_id
                )
            )
            image_data = fits.getdata(image_file, ext=0)
            images.append(image_data)
        images = np.asarray(images)
        images = np.reshape(images, (len(chunk), 101, 101, 1))
        # preprocess
        images = preprocess_image(images, train_mean, train_std)
        # Save ground truth
        y.append((chunk['no_source'] == 0).astype('int').values)
        # Predict
        predict = model.predict(images)
        y_pred.append(predict)
        y_pred_classes.append(proba_to_label(predict))
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    y_pred_classes = np.asarray(y_pred_classes)
    y = np.concatenate(y, axis=None)
    y_pred = np.concatenate(y_pred, axis=None)
    y_pred_classes = np.concatenate(y_pred_classes, axis=None)
    return y, y_pred, y_pred_classes
