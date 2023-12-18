from pca import PCA
import matplotlib.pyplot as plt
from utils import *

if __name__ == "__main__":
    img_path = "dima.jpeg"
    Y = []
    X = []
    for i in range(1, 502, 20):
        out_path = f"./trash/{i}_{img_path}"
        # comp_im = save_image_from_data(pca_transform(pca_compose(img_path=img_path), n_components=i), out_filename=out_path)
        data_img = image_info(img_path=out_path)
        X.append(data_img['image_size_kb'])
        Y.append(i)

    plt.plot(Y, X, color='b', label="size after compression with n components")
    plt.axhline(y=image_info(img_path=img_path)['image_size_kb'], color='r', linestyle='-', label="original size")
    plt.legend(loc="upper right")
    plt.xlabel('n components')
    plt.ylabel('size, KB')
    plt.show()
