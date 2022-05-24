#!/usr/bin/bash

gpu=1
num_classes=2
self_weight_path='../SSLModel/Reuslts/pretrained_weights/2022-01-13_23-08-05/ISIC_Unsup.pt'
num_epochs=10
batch_size=10
pre_train_type=None
lr=0.0001
loss=ce_dice
select_type=select
select_num=300
train_num=1600
iteration_num=9
active_epochs=2
aug_samples=35
load_idx='../overall_result/random_select_index/2022-01-17_20-04-08/index.npy'
round=10


cluster_idx_path_2048_10='../Cluster_Results/Hidden_feature/euclidean_AAP_2048_cluster10/2022-01-18_18-43-38_cluster_ids_x.pt'
cluster_npy_path_2048_10='../Cluster_Results/Hidden_feature/euclidean_AAP_2048_cluster10/2022-01-18_18-43-38_cluster.npy'


python train.py --Adam --gpu $gpu --num_classes $num_classes --num-epochs $num_epochs --batch-size $batch_size --pre_train_type None --lr $lr --loss $loss --select_type $select_type --select_num $select_num --train_num $train_num --iteration_num $iteration_num --active_epochs $active_epochs --aug_samples $aug_samples --load_idx $load_idx --round $round --weight_path $self_weight_path --cluster_idx_path $cluster_idx_path_2048_10 --cluster_npy_path $cluster_npy_path_2048_10
