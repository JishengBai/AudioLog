#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import math
from collections import OrderedDict
import soundfile as sf


class Encoder:
    def __init__(self, labels, audio_len, frame_len, frame_hop, net_pooling=1, sr=16000):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.sr = sr
        self.net_pooling = net_pooling
        n_samples = self.audio_len * self.sr
        self.n_frames = int(math.ceil(n_samples/2/self.frame_hop)*2 / self.net_pooling)

    def _time_to_frame(self, time):
        sample = time * self.sr
        frame = sample / self.frame_hop
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)

    def _frame_to_time(self, frame):
        time = frame * self.net_pooling * self.frame_hop / self.sr
        return np.clip(time, a_min=0, a_max=self.audio_len)

    def encode_strong_df(self, events_df):
        # from event dict, generate strong label tensor sized as [n_frame, n_class]
        true_labels = np.zeros((self.n_frames, len(self.labels)))
        for _, row in events_df.iterrows():
            if not pd.isna(row['event_label']):
                label_idx = self.labels.index(row["event_label"])
                onset = int(self._time_to_frame(row["onset"]))           #버림 -> 해당 time frame에 걸쳐있으면 true
                offset = int(np.ceil(self._time_to_frame(row["offset"])))  #올림 -> 해당 time frame에 걸쳐있으면 true
                true_labels[onset:offset, label_idx] = 1
        return true_labels

    def encode_weak(self, events):
        # from event dict, generate weak label tensor sized as [n_class]
        labels = np.zeros((len(self.labels)))
        if len(events) == 0:
            return labels
        else:
            for event in events:
                labels[self.labels.index(event)] = 1
            return labels

    def decode_strong(self, outputs):
        #from the network output sized [n_frame, n_class], generate the label/onset/offset lists
        pred = []
        for i, label_column in enumerate(outputs.T):  #outputs size = [n_class, frames]
            change_indices = self.find_contiguous_regions(label_column)
            for row in change_indices:
                onset = self._frame_to_time(row[0])
                offset = self._frame_to_time(row[1])
                onset = np.clip(onset, a_min=0, a_max=self.audio_len)
                offset = np.clip(offset, a_min=0, a_max=self.audio_len)
                pred.append([self.labels[i], onset, offset])
        return pred

    def decode_weak(self, outputs):
        result_labels = []
        for i, value in enumerate(outputs):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def find_contiguous_regions(self, array):
        #find at which frame the label changes in the array
        change_indices = np.logical_xor(array[1:], array[:-1]).nonzero()[0]
        #shift indices to focus the frame after
        change_indices += 1
        if array[0]:
            #if first element of array is True(1), add 0 in the beggining
            #change_indices = np.append(0, change_indices)
            change_indices = np.r_[0, change_indices]
        if array[-1]:
            #if last element is True, add the length of array
            change_indices = np.r_[change_indices, array.size]
        #reshape the result into two columns
        return change_indices.reshape((-1, 2))

def get_encoder(LabelDict, frame_length, hop_length, net_subsample, sr, audio_len):
    return Encoder(list(LabelDict.keys()),
                   audio_len=audio_len,
                   frame_len=frame_length,
                   frame_hop=hop_length,
                   net_pooling=net_subsample,
                   sr=sr)

def get_labeldict():
    return OrderedDict({"(object) banging": 0,
                        "(object) impact": 1,
                        "(object) rustling": 2,
                        "(object) snapping": 3,
                        "(object) squeaking": 4,
                        "bird singing": 5,
                        "brakes squeaking": 6,
                        "breathing": 7,
                        "car": 8,
                        "car passing by": 8,
                        "children": 9,
                        "children shouting": 9, 
                        "cupboard": 10,
                        "cutlery": 11,
                        "dishes": 12,
                        "drawer": 13,
                        "fan": 14,
                        "glass jingling": 15,
                        "keyboard typing": 16,
                        "large vehicle": 17,
                        "mouse clicking": 18,
                        "mouse wheeling": 19,
                        "object impact": 1,
                        "people speaking": 20,
                        "people talking": 20,
                        "people walking": 21,
                        "washing dishes": 22,
                        "water tap running": 23,
                        "wind blowing": 24
                        })

def get_csv(train_tsv, val_tsv, test_tsv):

    train_df = pd.read_csv(train_tsv, sep="\t")
    val_df = pd.read_csv(val_tsv, sep="\t")
    test_df = pd.read_csv(test_tsv, sep="\t")
    
    return train_df, val_df, test_df

def waveform_modification(filepath, pad_to, encoder):
    wav, _ = sf.read(filepath)
    wav = to_mono(wav)
    wav = pad_wav(wav, pad_to, encoder)

    return wav

def to_mono(wav, rand_ch=False):
    if wav.ndim > 1:
        if rand_ch:
            ch_idx = np.random.randint(0, wav.shape[-1] - 1)
            wav = wav[:, ch_idx]
        else:
            wav = np.mean(wav, axis=-1)
    return wav

def pad_wav(wav, pad_to, encoder):
    if len(wav) < pad_to:
        wav = np.pad(wav, (0, pad_to - len(wav)), mode="constant")
    else:
        wav = wav[:pad_to]
    return wav

class StronglyLabele(object):
    def __init__(self, tsv_read, dataset_dir, return_name, encoder):
        
        self.dataset_dir = dataset_dir
        self.encoder = encoder
        self.pad_to = encoder.audio_len * encoder.sr
        self.return_name = return_name

        #construct clip dictionary with filename = {path, events} where events = {label, onset and offset}
        clips = {}
        for _, row in tsv_read.iterrows():
            if row["path"] not in clips.keys():
                clips[row["path"]] = {"path": os.path.join(dataset_dir, row["path"]), "events": [], "scenes": []}
            if not np.isnan(row["onset"]):
                clips[row["path"]]["events"].append({"event_label": row["event_label"],
                                                         "onset": row["onset"],
                                                         "offset": row["offset"]})
                clips[row["path"]]["scenes"].append({"scene_label": row["scene_label"]})
        self.clips = clips #dictionary for each clip
        self.clip_list = list(clips.keys()) # list of all clip names

    def get_item(self, idx):

        filename = self.clip_list[idx]
        clip = self.clips[filename]
        path = clip["path"]

        # get wav
        wav = waveform_modification(path, self.pad_to, self.encoder)

        # get labels
        events = clip["events"]
        if not len(events): #label size = [frames, nclass]
            label = np.zeros((self.encoder.n_frames, len(self.encoder.labels)))
        else:
            label = self.encoder.encode_strong_df(pd.DataFrame(events))
        label = label.transpose(0, 1)

        # return
        out_args = [wav, label, idx]
        if self.return_name:
            out_args.extend([filename, path])
        return out_args