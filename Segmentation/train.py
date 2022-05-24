import os
import torch
from data import load_dataset, random_select_data, repre_select_data
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
from model import UNet
import numpy as np
import torch.optim as optim
import logging
from tqdm import tqdm
import sys
import random
from configure import get_arguments
import time
from utils import losses
from utils.logger import get_cur_time,checkpoint_save
from utils.lr import adjust_learning_rate,cosine_rampdown
import binary as mmb
import pdb
from datetime import datetime
import pytz
from test import test

# =================================================
#         load data path and get arguments
# =================================================
timenow = datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')),'%Y-%m-%d_%H-%M-%S')
basedir = '../overall_result' # save path
if not os.path.exists(basedir):
    os.makedirs(basedir)

args = get_arguments()
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
# Random seed
if args.manual_seed is None:
    args.manual_seed = random.randint(1, 10000)
np.random.seed(args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manual_seed)
    
args.loss = args.loss.strip()
batch_size = args.batch_size
base_lr = args.lr
num_classes = args.num_classes
img_size = [256,256]
pre_train_type = args.pre_train_type.strip()
load_idx = args.load_idx

# make dir
logdir = os.path.join(basedir, 'logs', str(args.select_type)+'_'+str(args.select_num), timenow)
print(logdir)
savedir = os.path.join(basedir, 'checkpoints', str(args.select_type)+'_'+str(args.select_num), timenow)
print(savedir)
shotdir = os.path.join(basedir, 'snapshot',str(args.select_type)+'_'+str(args.select_num), timenow)
print(shotdir)

os.makedirs(logdir, exist_ok=False)
os.makedirs(savedir, exist_ok=False)
os.makedirs(shotdir, exist_ok=False)

writer = SummaryWriter(logdir)

logging.basicConfig(filename=shotdir+"/"+"snapshot.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.info(str(args))

# =================================================
#                 data loader
# =================================================
imgspath_train = args.data_path_train + '/Train_imgs.npy'
maskspath_train = args.data_path_train + '/Train_mask.npy'
dataset = load_dataset(imgspath_train,maskspath_train,[256,256])
total_num = len(dataset)
train_dataset = dataset[0:args.train_num] # 1600 refer to CEAL as unlabeled pool
#imgspath_test = '../data/test/imgs_test.npy'
#maskspath_test = '../data/test/imgs_mask_test.npy'
#test_set = load_dataset(imgspath_test,maskspath_test,[256,256])
test_set = dataset[args.train_num:total_num]
    
print('\nSize of training set: {}'.format(len(train_dataset)))
print('Size of test set: {}\n'.format(len(test_set)))
    
def train(train_set,weight_path,num_epochs,pre_train):
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True)
    model = UNet(n_channels=1, n_classes=num_classes).cuda()
# =================================================
#           load pre-trained weights
# =================================================
    # load pre-trained weights from self-supervised learning
    if pre_train == 'init': # load all weights except the last output layer
        pretext_model = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in pretext_model.items() if 'outc' not in k}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)   
        print('\nload init weight from: ',weight_path)
    elif pre_train == 'encoder':    
        pretext_model = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in pretext_model.items() if 'outc' not in k if 'up' not in k}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)   
        print('\nload encoder weight from: ',weight_path)
    elif pre_train == 'all':
        model.load_state_dict(torch.load(weight_path))
        print('\nload all weight from: ',weight_path)
        
    model.train()
    
    if args.SGD:
        optimizer = optim.SGD(model.parameters(), lr=base_lr,
                              momentum=0.9, weight_decay=0.0001)
    if args.Adam:
        optimizer = optim.Adam(model.parameters(), lr=base_lr)

    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    logging.info("{} iterations per epoch".format(len(trainloader)))
    print('{} iterations per expoch'.format(len(trainloader)))
# =================================================
#                 training
# =================================================
    iter_num = 0
    best_performance = 0.0
    performance = 0.0


    for epoch_num in range(num_epochs):
        print("Epoch %d / %d : " %(epoch_num+1,num_epochs))

        loss_epoch = 0
        dice_epoch = 0
        for i_batch, sampled_batch in enumerate(tqdm(trainloader)):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.type(torch.FloatTensor).cuda(), label_batch.type(torch.LongTensor).cuda()

            outputs = model(volume_batch)

            loss_ce = ce_loss(outputs, label_batch)
            loss_dice, _ = dice_loss(outputs, label_batch)

            if args.loss == 'ce':
                loss = loss_ce
            elif args.loss == 'dice':
                loss = loss_dice
            elif args.loss == 'ce_dice':
                loss = 0.5 * (loss_dice + loss_ce)

            dice_epoch += 1 - loss_dice.item()
            loss_epoch += loss.item()

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

        epoch_dice = dice_epoch / len(trainloader)
        epoch_loss = loss_epoch / len(trainloader)
        writer.add_scalar('info/loss', epoch_loss, epoch_num +1)
        writer.add_scalar('info/dice', epoch_dice, epoch_num +1)

        logging.info('epoch %d : loss : %f dice: %f' % (epoch_num+1, epoch_loss, epoch_dice))
        print('epoch {} : loss : {} dice: {}'.format(epoch_num+1, epoch_loss, epoch_dice))

        checkpoint_path = checkpoint_save(model, True, savedir)
        logging.info("save model to {}".format(savedir))
    
    if args.Adam:
        writer.add_hparams({'log_dir':logdir, 'loss_func': args.loss,'optimizer': 'Adam', 'lr': args.lr, 'batch_size': args.batch_size, 'img_size':args.img_size,  'num_epoch':num_epochs}, {'val_dice': best_performance })
    elif args.SGD:
        writer.add_hparams({'log_dir':logdir, 'loss_func': args.loss,'optimizer': 'SGD', 'lr': args.lr, 'batch_size': args.batch_size, 'img_size':args.img_size,  'num_epoch':num_epochs}, {'val_dice': best_performance })
    print('========================================')
    print(pre_train_type) 
    dice_mean = test(test_set, checkpoint_path)
    print('Test epoch {} : dice: {}'.format(epoch_num+1, dice_mean))
    return checkpoint_path, dice_mean

def main():
    for round_idx in range(args.round):
        print('\n\n====================== round {} ==========================\n\n'.format(round_idx))
        dice_mean_list = []
        
        # criteria of inital sample selection
        if args.select_type == 'random':
            if args.load_idx == 'create': # random select
                random_index = random.sample(range(0,len(train_dataset)),len(train_dataset))
                idx_path = os.path.join(basedir,'random_select_index', timenow)
                os.makedirs(idx_path, exist_ok=False)
                load_idx = idx_path+'/index.npy'
                np.save(load_idx,random_index)
                print('Create random load index at:',load_idx)
            else: # use a determined list
                load_idx = args.load_idx
                print('random load index from: ',args.load_idx)
                
            train_set = random_select_data(train_dataset,args.select_num,load_idx)

        # select by clustering results
        elif args.select_type == 'select':
            train_set = repre_select_data(train_dataset,args.select_num,args.cluster_npy_path,args.cluster_idx_path)
        print('Train_set size: {} '.format(len(train_set)))
        print((train_set[0]['image']).shape,(train_set[0]['label']).shape)


        print('\n\n-------------- initialization training -----------------\n\n')
        select_num = args.select_num
        if pre_train_type == 'load':
            weight,dice_mean = train(train_set,args.weight_path,args.num_epochs,'init')
        elif pre_train_type == 'continue':
            weight,dice_mean = train(train_set,args.weight_path,args.num_epochs,'all')
        elif pre_train_type == 'encoder':
            weight,dice_mean = train(train_set,args.weight_path,args.num_epochs,'encoder')
        else:
            weight,dice_mean = train(train_set,args.weight_path,args.num_epochs,None)
        dice_mean_list.append(dice_mean)
        for i in range(args.iteration_num):
            print('\n-------------- training iteration {} -----------------\n\n'.format(i+1))
            select_num = select_num + args.aug_samples
            if args.select_type == 'random':
                train_set = random_select_data(train_dataset,select_num,load_idx)
            elif args.select_type == 'select':
                 train_set = repre_select_data(train_dataset,select_num,args.cluster_npy_path,args.cluster_idx_path)
            print('Train_set size: {} '.format(len(train_set)))
            print((train_set[0]['image']).shape,(train_set[0]['label']).shape)

            weight,dice_mean = train(train_set,weight,args.active_epochs,'all')
            dice_mean_list.append(dice_mean)
        
        cluster_path = args.cluster_npy_path
        cluster_path = cluster_path.split('/')[-2]
        list_path = os.path.join(args.result_path,str(args.select_type)+'_'+str(args.select_num)+'_'+pre_train_type,cluster_path)
        os.makedirs(list_path, exist_ok=True)
        np.save(os.path.join(list_path,str(round_idx)+'_'+timenow+'.npy'), dice_mean_list)
        print('save dice result path: ',os.path.join(list_path,str(round_idx)+'_'+timenow+'.npy'))
    writer.close()

if __name__ == "__main__":
    main()

    


