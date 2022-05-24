#!/usr/bin/env python
# coding: utf-8
# ref https://github.com/MrGiovanni/ModelsGenesis


import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
from torchsummary import summary
import sys
from utils import *
from unet_model2 import UNet
from config_cluster import models_genesis_config
from data_load import Dataset_Loader
import logging

print("torch = {}".format(torch.__version__))
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

conf = models_genesis_config()
conf.display()
img_size = [256,256]

# data load

'''
in unsupervised learning,
train set = 1600
validation set = 400
test 1: fix the train set for the first 1600
pixel value scale :[0,1]
resize: 16*N  = 256
'''
train_path = '../data/GrayData/Train_imgs.npy'
train_set = Dataset_Loader(train_path,img_size)

train_num =  1600
valid_num = 400
total_num = len(train_set)
x_train = train_set[0:train_num]
x_valid = train_set[train_num:train_num+valid_num]

logging.basicConfig(filename=conf.shotdir+"/"+"snapshot.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.info(str(conf))

print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))

training_generator = generate_pair(x_train,conf.batch_size, conf)
validation_generator = generate_pair(x_valid,conf.batch_size, conf)


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(n_channels=1, n_classes=conf.nb_class).cuda()
model.to(device)

print("Total CUDA devices: ", torch.cuda.device_count())

summary(model, (1,conf.input_rows,conf.input_cols), batch_size=-1)
criterion = nn.MSELoss()

if conf.optimizer == "sgd":
	optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
elif conf.optimizer == "adam":
	optimizer = torch.optim.Adam(model.parameters(), conf.lr)
else:
	raise

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(conf.patience * 0.8), gamma=0.5)

# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []
#best_loss = 100000
best_loss = 0.02
intial_epoch =0
num_epoch_no_improvement = 0
sys.stdout.flush()

if conf.weights != None:
	checkpoint=torch.load(conf.weights)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	intial_epoch=checkpoint['epoch']
	print("Loading weights from ",conf.weights)
sys.stdout.flush()


for epoch in range(intial_epoch,conf.nb_epoch):
    scheduler.step(epoch)
    model.train()
    for iteration in range(int(x_train.shape[0]//conf.batch_size)):
        image, gt = next(training_generator)
        gt = np.repeat(gt,conf.nb_class,axis=1)
        image,gt = torch.from_numpy(image).float().to(device), torch.from_numpy(gt).float().to(device)
        pred=model(image)
        pred=torch.sigmoid(pred)
        
        loss = criterion(pred,gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(round(loss.item(), 2))
        if (iteration + 1) % 5 ==0:
            print('Epoch [{}/{}], iteration {}, Loss: {:.6f}'.format(epoch + 1, conf.nb_epoch, iteration + 1, np.average(train_losses)))
        sys.stdout.flush()

    with torch.no_grad():
        model.eval()
        print("validating....")
        for i in range(int(x_valid.shape[0]//conf.batch_size)):
            x,y = next(validation_generator)
            y = np.repeat(y,conf.nb_class,axis=1)
            image,gt = torch.from_numpy(x).float(), torch.from_numpy(y).float()
            image=image.to(device)
            gt=gt.to(device)
            pred=model(image)
            pred=torch.sigmoid(pred)
            loss = criterion(pred,gt)
            valid_losses.append(loss.item())
    
    #logging
    train_loss=np.average(train_losses)
    valid_loss=np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
    train_losses=[]
    valid_losses=[]
    if valid_loss < best_loss:
        print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
        best_loss = valid_loss
        num_epoch_no_improvement = 0
        #save model
        torch.save({
            'epoch': epoch+1,
            'state_dict' : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },os.path.join(conf.model_path, "ISIC_Unsup.pt"))
        print("Saving model ",os.path.join(conf.model_path, "ISIC_Unsup.pt"))
    else:
        print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
        num_epoch_no_improvement += 1
    if num_epoch_no_improvement == conf.patience:
        print("Early Stopping")
        break
    sys.stdout.flush()



