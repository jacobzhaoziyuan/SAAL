import torch
from torch.utils.data import DataLoader
from model import UNet
import numpy as np
from tqdm import tqdm
from utils import losses
import binary as mmb
import pdb
import pytz

def test(test_set,checkpoint_path):
    net = 'UNet'
    path =checkpoint_path
    
    num_classes = 2
    img_size = [256,256]
    testloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)

    model = UNet(n_channels=1, n_classes=num_classes).cuda()  # old version
    model = model.cuda()
    print('load model from', path)
    model.load_state_dict(torch.load(path))
    model.eval()
    
    dice_list = []
    dice_test = []
    
    pred_vol_lst  = [np.zeros((img_size[0],img_size[1])) for i in range(len(testloader.sampler))]
    label_vol_lst = [np.zeros((img_size[0],img_size[1])) for i in range(len(testloader.sampler))]
    
    for i_batch, sampled_batch in enumerate(tqdm(testloader)):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.type(torch.FloatTensor).cuda(), label_batch.type(torch.LongTensor).cpu().numpy()

        outputs = model(volume_batch)
        outputs = torch.softmax(outputs, dim = 1)
        outputs = torch.argmax(outputs, dim = 1).cpu().numpy()
        
        pred_vol_lst[i_batch] = outputs[0,...]
        label_vol_lst[i_batch] = label_batch[0,...]

    for i in range(len(testloader)):
        dice_test.append(mmb.dc(pred_vol_lst[i], label_vol_lst[i]))
        for c in range(1, num_classes):   
            pred_test_data_tr = pred_vol_lst[i].copy()              
            pred_test_data_tr[pred_test_data_tr != c] = 0

            pred_gt_data_tr = label_vol_lst[i].copy() 
            pred_gt_data_tr[pred_gt_data_tr != c] = 0

            dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))

    dice_arr = 100 * np.reshape(dice_list, [-1, 2]).transpose()
    dice_mean = np.mean(dice_arr, axis=1)
    dice_std = np.std(dice_arr, axis=1)

    print('Dice0:',dice_mean[0])
    print('Dice1:',dice_mean[1])
    print('Dice test:',dice_arr.mean())
    return dice_arr.mean()