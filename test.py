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
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score

#from models import simple_net_v0

from tensorflow.keras.models import load_model

def data_preprocess_experimental_v0(path_dataset='/data/datasets/external/Adult_walking_v0/', sub_folders=['Fast','Slow','SlowPocket'],save=True):

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
#        dat,des = data_preprocess_experimental_v0(data_path,save=generate_merged_path)
        dat_path = data_path+'dat_merged.npy'
        des_path = data_path+'des_merged.csv'
        
        dat=np.load(dat_path)
        des=pd.read_csv(des_path,index_col=0)

        dat=dat[:,:,25:(25+176)]

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
    
    parser = argparse.ArgumentParser(description='Test function TFA v0')
    
    parser.add_argument('--test_data_path', type=str, default='/data/datasets/external/Adult_walking_v0/',
                        help='path to test data')
    parser.add_argument('--model_path', type=str, default='/data/models/model_dump/model_default_sim_v0_1598496538.h5',
                        help='path to model')
    parser.add_argument('--report_path', type=str, default='./',
                        help='path to report')
    parser.add_argument('--exp_type', type=str, default='exp_v0',
                        help='experiment type: sim_v0 or exp_v0')
    parser.add_argument('--report_name_tag', type=str, default='',
                        help='model name tag')
    parser.add_argument('--on_gpu', type=str, default='1',
                        help='gpu number')
   
    args = parser.parse_args()
 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.on_gpu
    
    [x_test,  y_test] = prepare_test(args.test_data_path, exp_type=args.exp_type)
        
    model = load_model(args.model_path)
     
    y_pred_test=model.predict(x_test[:,:,:])
#    y_pred_test_class=np.argmax(np.mean(y_pred_test,(1,2)),1)
    y_pred_test_class=np.argmax(y_pred_test,1)
    
    report_data={}
    
    conf_table_test=pd.crosstab(y_pred_test_class,y_test)
     
    report_file_path=args.report_path+'/'+args.report_name_tag+'_'+args.exp_type+'_'+str(int(time.time()))+'.json'
    
    conf_table_test=confusion_matrix(y_true=y_test,y_pred=y_pred_test_class)
    print("Test confusion matrix:\n")
    print(conf_table_test)
    accuracy=accuracy_score(y_test,y_pred_test_class)
    print("Validation accuracy:\n")
    print(accuracy)
    
    report_data['conf_table']=str(conf_table_test)
    report_data['accuracy']=str(accuracy)
        
    with open(report_file_path, 'w') as outfile:
        json.dump(report_data, outfile)

