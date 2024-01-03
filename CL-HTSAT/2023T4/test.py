import os
import sys
import torch.nn as nn
import torch
import torch.optim as optim

from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np

from model.htsat import HTSAT_Swin_Transformer
import config
from dataset import Test_Dataset
from dcase_util.containers import metadata
import sed_eval
from sklearn.metrics import accuracy_score
from utils import interpolate
import sed_scores_eval
import pandas as pd

# def label2class(idx_list, label_type):
#     for i in range(len(idx_list)):
#         if i == 0:
#             if label_type=='asc':
#                 class_str = config.asc_labels[idx_list[i]]
#             elif label_type=='sed':
#                 class_str = config.labels_hard[idx_list[i]]  
#         else:
#             if label_type=='asc':
#                 class_str = class_str+','+config.asc_labels[idx_list[i]]
#             elif label_type=='sed':
#                 class_str = class_str+','+config.labels_hard[idx_list[i]]  
#     return class_str
    
# def results2csv(output_folder, audio_name, asc_output, sed_output, time_res):
#     timestamp_arr = np.arange(0, len(sed_output) + 1) * time_res
#     onset_arr = []
#     offset_arr = []
#     scene_arr = []
#     event_arr = []

#     for i in range(len(timestamp_arr)-1):
#         if len(sed_output[i])!=0:
#             onset_arr.append(timestamp_arr[i])
#             offset_arr.append(timestamp_arr[i+1])
#             scene_arr.append(label2class(asc_output[i], 'asc'))
#             event_arr.append(label2class(sed_output[i], 'sed'))
    
#     result = {'onset':onset_arr, 'offset':offset_arr, 'scene':scene_arr, 'event':event_arr}
#     result_df = pd.DataFrame(result)
#     result_path = os.path.join(output_folder, audio_name+'.csv')
#     result_df.to_csv(result_path, index=False)
         
def label2class(idx, label_type):
    idx = int(idx)
    if label_type=='asc':
        class_str = config.asc_labels[idx]
    elif label_type=='sed':
        class_str = config.labels_hard[idx]  

    return class_str
    
def results2csv(output_folder, audio_name, asc_output, sed_output, time_res):
    timestamp_arr = np.arange(0, len(sed_output) + 1) * time_res
    onset_arr = []
    offset_arr = []
    scene_arr = []
    event_arr = []

    for i in range(len(timestamp_arr)-1):
        if len(sed_output[i])!=0:
            for j in sed_output[i]:
                onset_arr.append(timestamp_arr[i])
                offset_arr.append(timestamp_arr[i+1])
                scene_arr.append(label2class(asc_output[i], 'asc'))
                event_arr.append(label2class(j, 'sed'))
    
    result = {'onset(s)':onset_arr, 'offset(s)':offset_arr, 'scene':scene_arr, 'event':event_arr}
    result_df = pd.DataFrame(result)
    result_path = os.path.join(output_folder, audio_name+'.csv')
    result_df.to_csv(result_path, index=False)    
    
class ModelTester(object):
    def __init__(self,
                 model: nn.Module,
                 test_loader,
                 device: str,
                 ):
        '''
        :param model: torch model
        :param criterion: training loss function
        :param device: 'cpu' 'cuda1' 'cuda2'....
        :param valid_metric: 评价指标函数
        :param optimizer: 优化器
        '''
        self.device = self.get_device(device)
        self.model = model.to(self.device)
        self.test_data = test_loader
            
    def predict(self):
        self.model.eval()
        loader = tqdm(self.test_data)
        asc_output_list = []
        sed_output_list = []

        with torch.no_grad():
            for i, (data, ) in enumerate(loader):

                # load one batch of data
                data = data.reshape(data.shape[1], -1)
                data = data.to(self.device)

                output = self.model(data)
                
                asc_output = output['clipwise_output']  
                asc_output = torch.softmax(asc_output, dim=1)
                asc_output = torch.max(asc_output, 1)[1] 
                asc_output = asc_output.cpu().numpy()
                asc_output = asc_output.reshape(-1)

                sed_output = output['framewise_output']
                sed_output = sed_output[:, :250, :]
                sed_output = sed_output.cpu().numpy()
                sed_output = sed_output.reshape(-1, 11)

                asc_output_list.append(asc_output)
                sed_output_list.append(sed_output)

        asc_output_list = np.asarray(asc_output_list)
        sed_output_list = np.asarray(sed_output_list)
        
        return asc_output_list, sed_output_list
    
    def save_results(self, file_list, out_dir):
        asc_output, sed_output = self.predict()
        asc_output = np.repeat(asc_output, 10, axis=1)
        sed_output = sed_output[:, ::25, :]
        for idx, file_path in enumerate(file_list):
            file_name = file_path.split('/')[-1].split('.')[0]
            file_asc_output = asc_output[idx, :].reshape(-1, 1)
            file_sed_output = sed_output[idx, :]
            file_sed_idx = []
            for frame_sed in file_sed_output:
                sed_idx_arr = np.where(frame_sed>0.5)[0]
                file_sed_idx.append(sed_idx_arr)

            results2csv(config.exp_path, file_name, file_asc_output, file_sed_idx, 1)

    def get_device(self, device_id=None):
        '''
        :param device_id: 'cpu' 'cuda0' 'cuda1' ...
        :return:
        '''
        use_cuda = torch.cuda.is_available()
        if device_id is None:
            device = torch.device('cuda', 0)

        elif device_id == 'cpu':
            device = torch.device('cpu')

        elif len(device_id) == 4:
            device = torch.device('cuda', 0)

        elif len(device_id) > 4:
            assert device_id[:4] == 'cuda', ValueError
            device = torch.device(device_id[:4], int(device_id[-1]))
        return device

if __name__ == '__main__':
    
    os.makedirs(config.exp_path, exist_ok=True)
    
    ### data
    te_datagenerator = Test_Dataset(config.test_path, config.sample_rate, config.batch_size)
    te_loader = te_datagenerator.get_tensordataset()
    ### model
    model = HTSAT_Swin_Transformer(num_classes=config.classes_num, asc_num_classes=config.asc_classes_num, config=config)
    
    model_save_path = os.path.join(config.exp_path, config.exp_name)
    if config.resume_checkpoint is not None:
        ckpt = torch.load(model_save_path)
        model.load_state_dict(ckpt, strict=False)
        print('Loading params done!')
    
    ###
    predicter = ModelTester(model, te_loader, 'cuda1')
    
    # asc_output, sed_output = predicter.predict()
    test_file_list = te_datagenerator.file_list
    predicter.save_results(test_file_list, config.exp_path)
