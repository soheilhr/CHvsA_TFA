#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:43:24 2020

@author: Soheil
"""


import scipy.io as sio
from scipy.signal import spectrogram
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from models import simple_net_v0

def data_preprocess(path_dataset='/data/datasets/external/Adult_walking0/', sub_folders=['Fast','Slow','SlowPocket'],save=True):
    dat_all=[]
    des_all=[]
    for folder_name in sub_folders:
        path=path_dataset+folder_name
        dat_tmp=sio.loadmat(path+'/pre_processed.mat')['dat_all']
        des_tmp=pd.read_csv(path+'/pre_processed.csv')
#        des_tmp.Class=folder_name
        des_all.append(des_tmp)
        dat_all.append(dat_tmp)
    des_all=pd.concat(des_all)
    #des_all=np.concatenate(des_all)
    dat_all=np.concatenate(dat_all,2)
    dat_all=np.transpose(dat_all,(2,0,1))
    if save:
        des_all.to_csv(path_dataset+'des_merged.csv')
        np.save(path_dataset+'dat_merged.npy',dat_all)
    return dat_all,des_all

#dat,des=data_preprocess()

dat_path='/data/datasets/external/Adult_walking0/dat_merged.npy'
des_path='/data/datasets/external/Adult_walking0/des_merged.csv'

dat=np.load(dat_path)
des=pd.read_csv(des_path,index_col=0)

dat=np.concatenate([np.expand_dims(np.real(dat),-1),np.expand_dims(np.imag(dat),-1)],-1)
des.Class=des.Class.astype('category')

train_test_split=0.8
np.random.seed(0)
idxs=np.arange(dat.shape[0])
np.random.shuffle(idxs)
train_idx=idxs[0:int(train_test_split*dat.shape[0])]
test_idx=idxs[int(train_test_split*dat.shape[0]):]

x_train=dat[train_idx,...]
des_train=des.iloc[train_idx]
y_train=tf.keras.utils.to_categorical(des_train.Class.cat.codes)

x_test=dat[test_idx,...]
des_test=des.iloc[test_idx]
y_test=des_test.Class.cat.codes


input_shape=dat.shape[1:]
class_size=3

model=simple_net_v0(input_shape,class_size)
model.fit(x=x_train,y=y_train,epochs=32,batch_size=32)

y_pred=model.predict(x_test)
y_pred_class=np.argmax(y_pred,1)

conf_table=pd.crosstab(y_pred_class,y_test)



