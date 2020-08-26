#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:43:24 2020

@author: Soheil
"""


import scipy.io as sio
from scipy.signal import spectrogram
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import os
import time

from models import simple_net_v0




def data_preprocess_experimental_v0(path_dataset='/data/datasets/external/Adult_walking0/', sub_folders=['Fast','Slow','SlowPocket'],save=True):

    dat_all=[]
    des_all=[]

    for folder_name in sub_folders:
        path=path_dataset+folder_name

        dat_tmp=sio.loadmat(path+'/pre_processed.mat')['dat_all']
        des_tmp=pd.read_csv(path+'/pre_processed.csv')

        des_all.append(des_tmp)
        dat_all.append(dat_tmp)

    des_all=pd.concat(des_all)

    dat_all=np.concatenate(dat_all,2)
    dat_all=np.transpose(dat_all,(2,0,1))

    if save:
        des_all.to_csv(path_dataset+'des_merged.csv')
        np.save(path_dataset+'dat_merged.npy',dat_all)

    return dat_all,des_all

def data_preprocess_simulations(path_dataset='/data/datasets/simulations/Adult_walking_v0/',save=True):

    dat_all=[]
    
    dat=sio.loadmat(path_dataset+'pre_processed_specs_v0.mat')
    des=pd.read_csv(path_dataset+'pre_processed_labels_v0.csv')
    
    for spec in dat['x_specs']:
        dat_tmp=spec[0]
        dat_all.append(np.expand_dims(dat_tmp,0))

    dat_all=np.concatenate(dat_all,0)

    if save:
        des.to_csv(path_dataset+'des_merged.csv')
        np.save(path_dataset+'dat_merged.npy',dat_all)

    return dat_all,des



################ train and validation


def prepare_train_val(data_path, exp_type='sim_v0',from_merged_path=False, generate_merged_path=False):

    if from_merged_path:
        dat_path = data_path+'dat_merged.npy'
        des_path = data_path+'des_merged.csv'
        
        dat=np.load(dat_path)
        des=pd.read_csv(des_path,index_col=0)
    elif exp_type=='sim_v0':
        dat,des = data_preprocess_simulations(data_path,save=generate_merged_path)
    elif exp_type=='exp_v0':
        dat,des = data_preprocess_experimental_v0(data_path,save=generate_merged_path)
    else:
        print("Error, experiment type not understood\n input either sim_v0 or exp_v0")
        
    dat=np.concatenate([np.expand_dims(np.real(dat),-1),np.expand_dims(np.imag(dat),-1)],-1)
    des.Class=des.Class.astype('category')
    
    train_val_split=0.8
    np.random.seed(0)
    
    idxs=np.arange(dat.shape[0])
    np.random.shuffle(idxs)
    train_idx=idxs[:int(train_val_split*dat.shape[0])]
    val_idx=idxs[int(train_val_split*dat.shape[0]):]
    
    x_train=dat[train_idx,...]
    des_train=des.iloc[train_idx]
    y_train=tf.keras.utils.to_categorical(des_train.Class.cat.codes)
    y_train=np.expand_dims(y_train,(1,2))
    
    x_val=dat[val_idx,...]
    des_val=des.iloc[val_idx]
    y_val=des_val.Class.cat.codes
    
    return [x_train,y_train], [x_val,y_val]

################ test


def prepare_test(data_path, exp_type='exp_v0',from_merged_path=False, generate_merged_path=False):

    if from_merged_path:
        dat_path = data_path+'dat_merged.npy'
        des_path = data_path+'des_merged.csv'
        
        dat=np.load(dat_path)
        des=pd.read_csv(des_path,index_col=0)
    elif exp_type=='sim_v0':
        dat,des = data_preprocess_simulations(data_path,save=generate_merged_path)
    elif exp_type=='exp_v0':
        dat,des = data_preprocess_experimental_v0(data_path,save=generate_merged_path)
    else:
        print("Error, experiment type not understood\n input either sim_v0 or exp_v0")


    dat=np.concatenate([np.expand_dims(np.real(dat),-1),np.expand_dims(np.imag(dat),-1)],-1)
    des.Class=des.Class.astype('category')
    
    x_test=dat
    des_test=des
    y_test=des_test.Class.cat.codes
    return [x_test, y_test]

#####################################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training function TFA v0')
    
    parser.add_argument('--train_data_path', type=str, default='/data/datasets/simulations/Adult_walking_v0/',
                        help='path to train data')
    parser.add_argument('--model_save_path', type=str, default='/data/models/model_dump/',
                        help='path to save model')
    parser.add_argument('--exp_type', type=str, default='sim_v0',
                        help='experiment type: sim_v0 or exp_v0')
    parser.add_argument('--model_name_tag', type=str, default='',
                        help='model name tag')
    parser.add_argument('--on_gpu', type=str, default='1',
                        help='gpu number')
   
    args = parser.parse_args()
 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.on_gpu
    
    
    [x_train, y_train], [x_val, y_val] = prepare_train_val(args.train_data_path, exp_type=args.exp_type,from_merged_path=False, generate_merged_path=False)
    
    
    input_shape = x_train.shape[1:]
    class_size = y_train.shape[-1]
    
    model = simple_net_v0(input_shape,class_size)
    model.fit(x=x_train,y=y_train,epochs=32,batch_size=32,validation_split=0.2)

    y_pred_val=model.predict(x_val[:,:,:])
    y_pred_val_class=np.argmax(np.mean(y_pred_val,(1,2)),1)
    
    conf_table_val=pd.crosstab(y_pred_val_class,y_val)
    print("Validation confusion matrix:\n")
    print(conf_table_val)
    model_save_fname=args.model_save_path+'/model_'+args.model_name_tag+'_'+args.exp_type+'_'+str(int(time.time()))+'.h5'
    model.save(model_save_fname)

