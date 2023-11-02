import torch.utils.data as Data
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import scipy.io.wavfile as wav
import scipy.io
import librosa
import os
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset
from natsort import natsorted
from label_utils import get_csv, get_encoder, get_labeldict, StronglyLabele

class TUT_Dataset(object):
    def __init__(self, path:str, sr:int, batch_size:int, is_train:bool):

        self.path = path
        self.sr = sr
        self.batch_size = batch_size
        self.is_train = is_train
        self.file_list = []
        self.label_list = []
        self.get_file_list()
        
    def get_file_list(self):
        
        file_root_path = os.path.join(self.path, 'audio')
        file_name_list = os.listdir(file_root_path)
        file_name_list = natsorted(file_name_list)
        for file_name in file_name_list:
            self.file_list.append(os.path.join(file_root_path, file_name))
            
        label_root_path = os.path.join(self.path, 'label')
        label_name_list = os.listdir(label_root_path)
        label_name_list = natsorted(label_name_list)
        for label_name in label_name_list:
            self.label_list.append(os.path.join(label_root_path, label_name))

    def get_data(self):
        
        data = []
        label = []
        for file_path in self.file_list:
            file, sr = librosa.load(file_path, sr=None)
            file = librosa.resample(file, 44100, self.sr, res_type='kaiser_fast')
            if len(file)!=10*sr:
                file = np.concatenate((file, np.zeros(10*sr-len(file))), axis=0)
            data.append(file)
        for label_path in self.label_list:
            lb = np.load(label_path, allow_pickle=True)
            label.append(lb)

        data = np.asarray(data)    
        label = np.asarray(label)  
        
        return data, label
    
    def get_tensordataset(self):
        
        data, label = self.get_data()
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.from_numpy(label).float()
        dataset = TensorDataset(data_tensor, label_tensor) 
        if self.is_train:
            loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, 
                                pin_memory=True)
        else:
            loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, 
                                pin_memory=True)
        return loader
    
class Test_Dataset(object):
    def __init__(self, path:str, sr:int, batch_size:int):

        self.path = path
        self.sr = sr
        self.batch_size = batch_size
        self.file_list = []
        self.get_file_list()
        
    def get_file_list(self):

        file_name_list = os.listdir(self.path)
        file_name_list = natsorted(file_name_list)
        for file_name in file_name_list:
            self.file_list.append(os.path.join(self.path, file_name))

    def get_data(self):
        
        data = []
        for file_path in self.file_list:
            seg_arr = []
            file, sr = librosa.load(file_path, sr=32000, res_type='kaiser_fast')
            num_seg = len(file)//(sr*10)
            for num in range(num_seg):
                segment = file[num*sr*10:(num+1)*sr*10]
                seg_arr.append(segment)
            seg_arr = np.asarray(seg_arr)  
            data.append(seg_arr)
        data = np.asarray(data)
        
        return data
    
    def get_tensordataset(self):
        
        data = self.get_data()
        data_tensor = torch.from_numpy(data).float()
        dataset = TensorDataset(data_tensor) 
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=2, 
                                pin_memory=True)
        return loader

if __name__=='__main__':
    import config
    val_datagenerator = DCASE23T4B_Dataset(config.val_path, config.sample_rate, config.batch_size, False)
    val_dataloader = val_datagenerator.get_tensordataset()
    