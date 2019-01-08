# 用于将cifar10的数据可视化
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from PIL import Image
import os

def load_CIFAR_batch(filename):
    with open(filename, 'rb')as f:
        #       datadict = p.load(f)
        datadict = p.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


if __name__ == "__main__":
    if not os.path.exists("cifar10_images"):
        os.mkdir("cifar10_images")

    imgXD, imgY = load_CIFAR_batch(r"C:\Users\lmc\Desktop\cifar-10-batches-py\data_batch_1") # type: np.ndarray

    for i in range(len(imgXD)):
        if not os.path.exists(os.path.join("Face",f'{imgY[i]}')):
            os.mkdir(os.path.join("Face",f'{imgY[i]}'))
        imgs = imgXD[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB", (i0, i1, i2))
        name = str(i) + ".png"
        img.save(os.path.join("Face",f'{imgY[i]}', name) , "png")
