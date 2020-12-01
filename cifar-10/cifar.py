import os
import cv2 as cv
import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def write_image(img_raw, path):
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    blue = img_raw[:1024]
    green = img_raw[1024: 2048]
    red = img_raw[2048:]
    for i in range(32):
        for j in range(32):
            index = i * 32 + j
            img[i][j] = [red[index], green[index], blue[index]]
    cv.imwrite(path + ".jpg", img)


def write_batch(path_batch, labels):
    batch = unpickle(path_batch)
    for i in range(10000):
        img_raw = batch[b'data'][i]
        img_label = batch[b'labels'][i]
        img_index = len(os.listdir("../data/colectii/" + labels[img_label]))
        img_path = "../data/colectii/" + labels[img_label] + "/" + str(img_index)
        write_image(img_raw, img_path)


batch_meta = unpickle("batches.meta")
labels = [str(batch_meta[b'label_names'][i], "utf-8") for i in range(10)]
path = "../data/colectii/"
for i in range(10):
    os.mkdir(path + labels[i])
for i in range(1, 6):
    write_batch("data_batch_" + str(i), labels)
    print(f"done the {i} batch")
