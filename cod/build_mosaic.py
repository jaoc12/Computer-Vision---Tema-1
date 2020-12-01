import os
import matplotlib.pyplot as plt

from add_pieces_mosaic import *
from parameters import *


def load_pieces(params: Parameters):
    # citeste toate cele N piese folosite la mozaic din directorul corespunzator
    # toate cele N imagini au aceeasi dimensiune H x W x C, unde:
    # H = inaltime, W = latime, C = nr canale (C=1  gri, C=3 color)
    # functia intoarce pieseMozaic = matrice N x H x W x C in params
    # pieseMoziac[i, :, :, :] reprezinta piesa numarul i
    images_list = os.listdir(params.small_images_dir)
    images = []
    for it, img in enumerate(images_list):
        if params.color is True:
            images.append(cv.imread(params.small_images_dir + img))
        else:
            images.append(cv.imread(params.small_images_dir + img, cv.IMREAD_GRAYSCALE))
        print('Loading small images %.2f%%' % (100 * (it + 1) / len(images_list)))
    images = np.array(images, dtype=np.float32)

    # citeste imaginile din director

    if params.show_small_images:
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                # OpenCV reads images in BGR format, matplotlib reads images in RBG format
                im = images[i * 10 + j].copy()
                # BGR to RGB, swap the channels
                im = im[:, :, [2, 1, 0]]
                plt.imshow(im)
        plt.show()

    params.small_images = images


def compute_dimensions(params: Parameters):
    # calculeaza dimensiunile mozaicului
    # obtine si imaginea de referinta redimensionata avand aceleasi dimensiuni
    # ca mozaicul

    # completati codul
    # calculeaza automat numarul de piese pe verticala
    h, w, _ = params.image.shape
    if params.color is True:
        h_small, w_small, _ = params.small_images[0].shape
    else:
        h_small, w_small = params.small_images[0].shape
    # calculam numarul de piese necesar pentru a pastra raportul original dintre h si w
    params.num_pieces_vertical = np.int((w_small * h * params.num_pieces_horizontal) / (w * h_small))

    # redimensioneaza imaginea
    new_h = h_small * params.num_pieces_vertical
    new_w = w_small * params.num_pieces_horizontal
    params.image_resized = cv.resize(params.image, (new_w, new_h))
    if params.color is False:
        params.image_resized = cv.cvtColor(params.image_resized, cv.COLOR_BGR2GRAY)


def build_mosaic(params: Parameters):
    # incarcam imaginile din care vom forma mozaicul
    load_pieces(params)
    # calculeaza dimensiunea mozaicului
    compute_dimensions(params)

    img_mosaic = None
    if params.layout == 'caroiaj':
        if params.hexagon is True:
            img_mosaic = add_pieces_hexagon(params)
        else:
            img_mosaic = add_pieces_grid(params)
    elif params.layout == 'aleator':
        img_mosaic = add_pieces_random(params)
    else:
        print('Wrong option!')
        exit(-1)

    return img_mosaic
