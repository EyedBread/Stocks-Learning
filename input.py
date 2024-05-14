#!/usr/bin/env python
# coding: utf-8

import os, json
import pandas as pd
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from numpy import newaxis

split = (0.85);
sequence_length=10;
normalise= True
batch_size=100;
input_dim=5
input_timesteps=9
neurons=50
epochs=5
prediction_len=1
dense_output=1
drop_out=0


def load_data(file_path):
    """
    Load the data from the file path into training and testing data
    :param file_path: str: file path to the data
    :return: pd.DataFrame: the data in a pandas data frame
    """
    # Load the data
    dataframe = pd.read_csv(file_path)

    return dataframe


def partition_data(dataframe):
    """
    Partition the data into training and testing data, normalize it
    :param dataframe: pd.DataFrame: the data in a pandas data frame
    :return: tuple: the training and testing data
    """
    # Parse the data
    cols = ['Adj Close','wsj_mean_compound','cnbc_mean_compound','fortune_mean_compound',
          'reuters_mean_compound']
    i_split = int(len(dataframe) * split)
    data_train = dataframe.get(cols).values[:i_split]
    data_test  = dataframe.get(cols).values[i_split:]
    len_train  = len(data_train)
    len_test   = len(data_test)
    len_train_windows = None
    print('data_train.shape',data_train.shape)
    print('data_test.shape',data_test.shape)
    data_train[0:5]



    data_test[0:5]


    #get_test_data   
    data_windows = []
    for i in range(len_test - sequence_length):
        data_windows.append(data_test[i:i+sequence_length])
    data_windows = np.array(data_windows).astype(float)
    # get original y_test
    y_test_ori = data_windows[:, -1, [0]]
    print('y_test_ori.shape',y_test_ori.shape)

    window_data=data_windows
    win_num=window_data.shape[0]
    col_num=window_data.shape[2]
    normalised_data = []
    record_min=[]
    record_max=[]

    #normalize
    for win_i in range(0,win_num):
        normalised_window = []
        for col_i in range(0,1):#col_num):
            temp_col=window_data[win_i,:,col_i]
            temp_min=min(temp_col)
            if col_i==0:
                record_min.append(temp_min)#record min
            temp_col=temp_col-temp_min
            temp_max=max(temp_col)
            if col_i==0:
                record_max.append(temp_max)#record max
            temp_col=temp_col/temp_max
            normalised_window.append(temp_col)
        for col_i in range(1,col_num):
            temp_col=window_data[win_i,:,col_i]
            normalised_window.append(temp_col)
        normalised_window = np.array(normalised_window).T
        normalised_data.append(normalised_window)
    normalised_data=np.array(normalised_data)

    # normalised_data=window_data
    data_windows=normalised_data#get_test_data
    x_test = data_windows[:, :-1]
    y_test = data_windows[:, -1, [0]]

    return data_train, data_test, x_test, y_test, y_test_ori, record_min, record_max
    