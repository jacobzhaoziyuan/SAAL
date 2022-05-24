from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import random
from tqdm import tqdm
import torch


def preprocessor(input_img, img_cols, img_rows):
    """
    Resize input images to constants sizes
    :param input_img: numpy array of images
    :return: numpy array of preprocessed images
    """
    output_img = np.ndarray((input_img.shape[0], input_img.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(input_img.shape[0]):
        output_img[i, 0] = cv2.resize(input_img[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return output_img

def load_dataset(imgspath,maskspath,img_size):
    """
    Load training data from project path
    :return: [images, masks] numpy arrays containing the training data and their respective masks.
    """
    print("\nLoading data...\n")
    rows = img_size[0]
    cols = img_size[1]
    
    images = np.load(imgspath)
    masks = np.load(maskspath)
    images = preprocessor(images, cols, rows)
    masks = preprocessor(masks, cols, rows)

    images = images.astype('float32')
    masks = masks.astype('float32')
    masks /= 255.  # scale masks to [0, 1]

    dataset = []
    for idx in tqdm(range(len(images))):
        img = cv2.resize(images[idx][0,...],(cols,rows),interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(masks[idx][0,...],(cols,rows),interpolation=cv2.INTER_CUBIC)
        dataset.append({'image':img[np.newaxis,...],'label':label.astype(np.uint8)})
        
    print("images: {} | {:.2f} ~ {:.2f}".format(len(images), np.min(images), np.max(images)))
    print("masks: {} | {:.2f} ~ {:.2f}".format(len(masks), np.min(masks), np.max(masks)))
        
    return dataset

def random_select_data(train_dataset,select_num,load_idx):
    train_set = []    
    random_index = np.load(load_idx)
    print('\nRandom load index from: ',load_idx)
    for idx in range(select_num):
        train_set.append(train_dataset[idx])
    return train_set

def repre_select_data(train_dataset,select_num,cluster_npy_path,cluster_idx_path):
    # load clustering results
    dissort = np.load(cluster_npy_path,allow_pickle=True)
    discla = torch.load(cluster_idx_path)
    disdict = discla.cpu().detach().numpy()
    unique, counts = np.unique(disdict, return_counts=True)
    print(dict(zip(unique, counts)))
    num = [round(counts[i]/len(train_dataset)*select_num) for i in range(len(counts))]
    num[-1] = select_num-np.sum(num[0:-1])
    print(dict(zip(unique, num)))
    
    train_set = []
    for i in range(len(unique)): # cluster centers
        for idx in range(num[i]): # sample index
            index_list = dissort[i]['idx_sort']
            train_set.append(train_dataset[index_list[idx]])
    return train_set