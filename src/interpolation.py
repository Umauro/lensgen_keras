import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def slerp(p_0, p_1, t):
    """
        Retorna un punto entre otros dos puntos definidos utilizando
        interpolación linear esférica
        
        Atributos:
            - p_0: Numpy Array, primer punto en la interpolación
            - p_1: Numpy Array, segundo punto en la interpolación
            - t: Float entre 0 y 1. Ajusta que tan cerca estará el nuevo punto de p_0 o p_1
    """
    assert 0 <= t <= 1, "t debe estar entre 0 y 1"
    theta = np.arccos(cosine_similarity(p_0, p_1))  # angulo entre p_0 y p_1
    return (((np.sin(1 - t) * theta) / np.sin(theta)) * p_0) + ((np.sin(t * theta)) / (np.sin(theta))) * p_1


def interpolate_latent_space(g_model, latent_dim, rows, cols):
    """
        Genera una grilla de rows x cols imágenes interpolando
        el espacio latente utilizando SLERP
        
        Atributos:
            - g_model: Keras model. Generador de la gan
            - latent_dim: Int. Dimensión del espacio latente
    """
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for row in range(rows):
        p_0 = np.random.normal(0, 1, (1, latent_dim))  # random points
        p_1 = np.random.normal(0, 1, (1, latent_dim))
        for col, t in enumerate(np.linspace(0, 1, cols)):
            new_point = slerp(p_0, p_1, t)
            synthetic_image = g_model.predict(new_point)[0, :, :, 0]
            synthetic_image = synthetic_image * 0.5 + 0.5  # scale between 0 and 1
            axes[row][col].imshow(synthetic_image, cmap='hot')
            axes[row][col].axis('off')
    plt.show()
