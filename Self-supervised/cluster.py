#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
from torchsummary import summary
import sys
from utils import *
# import unet3d
from unet_model import UNet, UNet_hidden
from config_cluster import models_genesis_config
from tqdm import tqdm
from data_load import Dataset_Loader
from datetime import datetime
import logging

import numpy as np
import matplotlib.pyplot as plt
from kmeans_func import kmeans, kmeans_predict
import os
from datetime import datetime
import pytz

# =================================================
#             load data and model
# =================================================

print("torch = {}".format(torch.__version__))
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Total CUDA devices: ", torch.cuda.device_count())
img_size = [256,256]
input_rows, input_cols = 256, 256
conf = models_genesis_config()

train_path = './data/Train_imgs.npy'
train_set = Dataset_Loader(train_path,img_size)

train_num =  1600
x_train = train_set[0:train_num]
print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
training_generator = generate_pair(x_train,conf.batch_size, conf)

model = UNet_hidden(n_channels=1, n_classes=conf.nb_class).cuda()
model.to(device)
summary(model, (1,input_rows,input_cols), batch_size=-1)
criterion = nn.MSELoss()

# =================================================
#             extract hidden features
# =================================================
conf.weights = '../SSLModel/Reuslts/pretrained_weights/2022-01-13_23-08-05/ISIC_Unsup.pt'

if conf.weights != None:
    checkpoint=torch.load(conf.weights)
    model.load_state_dict(checkpoint['state_dict'])
    print("Loading weights from ",conf.weights)
sys.stdout.flush()

feature_map = []
for iteration in tqdm(range(int(x_train.shape[0]//conf.batch_size))):
    image, _ = next(training_generator)
    image = torch.from_numpy(image).float().to(device)
    _, feature=model(image)
    descriptors = feature.cpu().detach().numpy()
    for i in range(conf.batch_size):
        feature_map.append(descriptors[i])
print('\nsize of feature_map:',np.shape(feature_map))
np.save(os.path.join(conf.model_path,'2022-01-13_23-08-05_feature_map.npy'),feature_map)
print('path of feature map:',os.path.join(conf.model_path,'2022-01-13_23-08-05_feature_map.npy'))
'''
feature_path = '../SSLModel/Reuslts/pretrained_weights/2022-01-13_04-14-30/feature_map.npy'
feature_map = np.load(feature_path)
'''
newmap=torch.from_numpy(np.array(feature_map))

# =================================================
#           dimensionality reduction
# =================================================
class PCA(object):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return X.matmul(self.proj_mat)

print('========== processing dimensionality reduction ===========')
redim_type = 'pooling_512'
dim = 512
if redim_type == 'flatten_PCA':  # in paper, no use PCA
    # flatten
    flatten_map = torch.flatten(newmap, start_dim=1, end_dim=-1)
    # pca
    pca = PCA(n_components=np.shape(flatten_map)[1])
    pca.fit(flatten_map)
    X_all = pca.transform(flatten_map)
    reduced = X_all[:,0:dim]
    
elif redim_type == 'pooling_512':
    # adaptive average pool
    aap512 = nn.AdaptiveAvgPool2d((1))
    map_aap512 = aap512(newmap)
    reduced =  torch.flatten(map_aap512, start_dim=1, end_dim=-1)
    
elif redim_type == 'pooling_2048':
    aap2048 = nn.AdaptiveAvgPool2d((2,2))
    map_aap2048 = aap2048(newmap)
    reduced = torch.flatten(map_aap2048, start_dim=1, end_dim=-1)
    
print(redim_type)
print('\nfeature shape:', np.shape(reduced))

# =================================================
#                     clustering
# =================================================
dir_path = '../Cluster_Results/Hidden_feature' # save path of features
matrix = 'euclidean'
num_clusters = 10

# dim_list = [512,256,128,64]
dim_list = 512
for dim in dim_list:
    # reduced = X_all[:,0:dim]
    x = reduced
    
    name = matrix+'_'+redim_type+'_dim'+str(dim)
    save_path = os.path.join(dir_path,name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('/nFeatures save under: ',save_path)

    timenow = datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')),'%Y-%m-%d_%H-%M-%S')


    cluster_ids_x, cluster_centers = kmeans(
        X=x, num_clusters=num_clusters, distance=matrix, device=device
    )
    cluster_dis = kmeans_predict(x, cluster_centers, matrix, device=device)

    print('\nsave at:')
    print('cluster_ids_x :',os.path.join(save_path,timenow + '_cluster_ids_x.pt'))
    print('cluster_centers: ', os.path.join(save_path,timenow + '_cluster_centers.pt'))
    print('cluster_distances: ', os.path.join(save_path,timenow + '_cluster_centers_dis.pt'))

    torch.save(cluster_ids_x,os.path.join(save_path,timenow + '_cluster_ids_x.pt'))
    torch.save(cluster_centers,os.path.join(save_path,timenow + '_cluster_centers.pt'))    
    torch.save(cluster_dis,os.path.join(save_path,timenow + '_cluster_centers_dis.pt'))

    cluster_map = []
    for i in range(num_clusters):
        dis, idx_sort = torch.sort(cluster_dis[:,i], dim=0, descending=False)
        cluster_map.append({'dis':dis,'idx_sort':idx_sort})

    print('cluster_distance rank: ', os.path.join(save_path,timenow + '_cluster.npy'))
    np.save(os.path.join(save_path,timenow + '_cluster.npy'),cluster_map)
