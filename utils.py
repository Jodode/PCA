from PIL import Image
import numpy as np
import os
from pca import PCA

def image_info(img_path: str):
    original_image = Image.open(img_path)

    imgage_size_kb = os.stat(img_path).st_size/1024
    data = original_image.getdata()

    original_pixels = np.array(data).reshape(*original_image.size, -1)
    image_dimensions = original_pixels.shape

    return {'image_size_kb': imgage_size_kb, 'image_dimensions': image_dimensions}

def image_to_matrix(img_path: str):
    original_image = Image.open(img_path)

    data = np.array(original_image.getdata())
    data = data.reshape(*original_image.size, -1)
    
    return data

def pca_compose(img_path: str):
    pca_channel = {}

    data = image_to_matrix(img_path)

    transposed_data = np.transpose(data)
    for i in range(data.shape[-1]):
        channel = transposed_data[i].reshape(*data.shape[:-1])

        pca = PCA()
        fit_pca = pca.fit_transform(channel)

        pca_channel[i] = (pca, fit_pca)
    
    return pca_channel

def pca_transform(pca_channel, n_components):
    temp_image_result = []

    for channel in range(len(pca_channel)):
        pca, fit_pca = pca_channel[channel]

        pca_pixels = fit_pca[:, :n_components]
        
        pca_components = pca.components_[:n_components, :]

        compressed_pixels = np.dot(pca_pixels, pca_components) + pca.mean_

        temp_image_result.append(compressed_pixels)

    compressed_image = np.transpose(temp_image_result)
    compressed_image = np.array(compressed_image,dtype=np.uint8)

    return compressed_image

def save_image_from_data(compressed_image, out_filename: str):
    im = Image.fromarray(compressed_image)
    im.save(out_filename)