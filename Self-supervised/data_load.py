from torch.utils.data import Dataset
import os
import numpy as np
import re
import imgaug.augmenters as iaa
#from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torchvision import transforms
import cv2
import random
from tqdm import tqdm

import matplotlib.pyplot as plt

random.seed(1)


def Dataset_Loader(path, img_size):
    print('\nLoading dataset...\n')
    read_imgs = np.load(path)
    rows = img_size[0]
    cols = img_size[1]
    
    images = np.ndarray((read_imgs.shape[0], read_imgs.shape[1], rows, cols), dtype=np.float)
    for i in range(read_imgs.shape[0]):
        img = cv2.resize(read_imgs[i, 0], (cols, rows), interpolation=cv2.INTER_CUBIC)
        images[i, 0, :, :] = img/255.
    return images

if __name__ == "__main__":
    data_path_train = '../data/GratData'
    trainpath = data_path_train + '/Train_imgs.npy'
    
    dataset = Dataset_Loader(trainpath,[256,256])
