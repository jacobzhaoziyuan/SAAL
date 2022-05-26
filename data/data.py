from __future__ import print_function

import os
import numpy as np
from random import shuffle
import cv2

from constants import *

random.seed(0)


def preprocessor(input_img):
    """
    Resize input images to constants sizes
    :param input_img: numpy array of images
    :return: numpy array of preprocessed images
    """
    print(input_img.shape)
    output_img = np.ndarray((input_img.shape[0],img_rows, img_cols, input_img.shape[3]), dtype=np.uint8)
    for i in range(input_img.shape[0]):
        # output_img[i, 0] = cv2.resize(input_img[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        output_img[i,:,:,0] = cv2.resize(input_img[i,:,:,0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        # print(output_img[i].shape)
    return output_img


def create_train_data():
    """
    Generate training data numpy arrays and save them into the project path
    """
    if os.path.exists(global_path + 'imgs_train.npy') and os.path.exists(global_path + 'imgs_mask_train.npy'):
        return

    image_rows = img_rows
    image_cols = img_cols

    images = os.listdir(data_path)
    masks = os.listdir(masks_path)
    total = len(images)

    # imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    print(imgs.shape)
    # imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    print(imgs_mask.shape)
    i = 0
    shuffle(images)
    for image_name in images:
        print(i)
        print(image_name)
        img = cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
        img = np.array([img])
        print('img shape before :',img.shape)
        img = np.rollaxis(img, 0,3)
        print('img shape after :',img.shape)
        imgs[i] = img

        # img_mask = cv2.imread(os.path.join(masks_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(masks_path, image_name.replace('.jpg', '_segmentation.png')), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.resize(img_mask, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
        img_mask = np.array([img_mask])
        # img_mask = np.reshape(img_mask, (image_rows, image_cols, 1))
        img_mask = np.rollaxis(img_mask, 0,3)
        print('img_mask shape: ', img_mask.shape)
        imgs_mask[i] = img_mask
        i += 1

    np.save(global_path + 'imgs_train.npy', imgs)
    np.save(global_path + 'imgs_mask_train.npy', imgs_mask)


def create_val_data(sig="val"):
    """
    Generate validation/testing data numpy arrays and save them into the project path
    """
    if os.path.exists(global_path + f'imgs_{sig}.npy') and os.path.exists(global_path + f'imgs_mask_{sig}.npy'):
        return

    image_rows = img_rows
    image_cols = img_cols

    images = os.listdir(val_data_path.replace("test", sig) if sig != "test" else val_data_path)
    total = len(images)

    # imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    print(imgs.shape)
    imgs_mask = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    print(imgs_mask.shape)
    i = 0
    shuffle(images)
    for image_name in images:

        img = cv2.imread(os.path.join(val_data_path.replace("test", sig) if sig != "test" else val_data_path,
                                      image_name), cv2.IMREAD_GRAYSCALE)
        print(image_name)
        img = cv2.resize(img, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
        img = np.array([img])
        img = np.rollaxis(img, 0,3)
        imgs[i] = img

        img_mask = cv2.imread(os.path.join(val_data_path.replace("test", sig) if sig != "test" else val_data_path,
                                           image_name.replace('.jpg', '_segmentation.png')), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.resize(img_mask, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
        img_mask = np.array([img_mask])
        # img_mask = np.reshape(img_mask, (image_rows, image_cols, 1))
        img_mask = np.rollaxis(img_mask, 0,3)
        imgs_mask[i] = img_mask
        i += 1

    np.save(global_path + f'imgs_{sig}.npy', imgs)
    np.save(global_path + f'imgs_mask_{sig}.npy', imgs_mask)


def load_train_data():
    """
    Load training data from project path
    :return: [X_train, y_train] numpy arrays containing the training data and their respective masks.
    """
    print("\nLoading train data...\n")
    X_train = np.load(global_path + 'imgs_train.npy')
    y_train = np.load(global_path + 'imgs_mask_train.npy')

    X_train = preprocessor(X_train)
    y_train = preprocessor(y_train)

    X_train = X_train.astype('float32')

    if z_score:
        mean = np.mean(X_train)  # mean for data centering
        std = np.std(X_train)  # std for data normalization

        X_train -= mean
        X_train /= std
    else:
        X_train /= 255.

    y_train = y_train.astype('float32')
    y_train /= 255.  # scale masks to [0, 1]
    return X_train, y_train


def unnormalize(img: np.array):
    # img: [H, W, C]
    img = img.reshape([img_rows, img_cols, 1])
    img = 255 * (img - img.min()) / (img.max() - img.min())
    return img


def load_val_data(sig="val"):
    """
    Load training data from project path
    :return: [X_train, y_train] numpy arrays containing the training data and their respective masks.
    """
    print(f"\nLoading {sig} data...\n")
    X_val = np.load(global_path + f'imgs_{sig}.npy')
    y_val = np.load(global_path + f'imgs_mask_{sig}.npy')

    X_val = preprocessor(X_val)
    y_val = preprocessor(y_val)

    X_val = X_val.astype('float32')

    if z_score:
        mean = np.mean(X_val)  # mean for data centering
        std = np.std(X_val)  # std for data normalization

        X_val -= mean
        X_val /= std
    else:
        X_val /= 255.

    y_val = y_val.astype('float32')
    y_val /= 255.  # scale masks to [0, 1]
    return X_val, y_val


if __name__ == '__main__':
    create_train_data()
