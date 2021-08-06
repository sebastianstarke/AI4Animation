'''
Author : Yiwei Zhao
Date : Sep 18, 2019
'''
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import re

import pandas as pd
import os
import numpy as np
import random

from scipy.ndimage import gaussian_filter1d


class get_smooth(object):
    def __init__(self, list=(), sigma=(1.0, 1.0), inplace=False):
        self.list = list
        self.sigma = sigma
        self.inplace = inplace

    def __call__(self, sample):
        for name in self.list:
            rand_sig = np.random.uniform(self.sigma[0], self.sigma[1])
            if self.inplace:
                sample[name] = gaussian_filter1d(sample[name], rand_sig, axis=0)
            else:
                sample[name+'_smooth'] = gaussian_filter1d(sample[name], rand_sig, axis=0)

        return sample


class sub_average(object):
    def __init__(self, average, std, cutoff_channel=False):
        self.average = average
        self.std = std
        self.cutoff_channel = cutoff_channel


    def __call__(self, sample):
        for key in self.average.keys():
            if self.cutoff_channel:
                sample[key] -= self.average[key][:sample[key].shape[1]]
                sample[key] = np.true_divide(sample[key], self.std[key][:sample[key].shape[1]])
            else:
                sample[key] -= self.average[key]
                sample[key] = np.true_divide(sample[key], self.std[key])
        return sample


class ToTensor(object):
    def __init__(self, batch=False, list=('input',)):
        self.batch = batch
        self.list = list

    def __call__(self, sample):
        tensor_list = {}
        for name in self.list:
            # swap length aixs
            # numpy image: L X C
            # torch image: C X L
            input_m = sample[name].transpose((1, 0))
            if self.batch:
                input_m = input_m.reshape((1, input_m.shape[0], input_m.shape[1]))
            tensor_list[name] = torch.from_numpy(input_m)
        return tensor_list


def get_time_channel(label):
    if len(re.findall(r'\d+', label)) < 1:
        return (None, None)
    time = re.findall(r'\d+', label)[0]
    name = label[:label.find(time)] + label[label.find(time) + len(time):]
    return time, name


class ControlDataset(Dataset):
    """Control dataset."""

    def parse_label(self, label_file):
        self.channel = {}
        self.id_to_time_channel = {}

        f = open(label_file, "r")
        for x in f:
            id, name = x.split('] ')[0][1:], x.split('] ')[1].replace('\n', '')

            if 'Player' in name or 'Gating' in name:
                break

            time, channel_name = get_time_channel(name)
            if time is None:
                continue

            if not channel_name in self.channel.keys():
                self.channel[channel_name] = len(self.channel)

            self.id_to_time_channel[int(id)] = (time, self.channel[channel_name])

        self.channel_num = len(self.id_to_time_channel.keys())

    def parse_data(self, txt_file):
        f = open(txt_file, "r")
        data_array = []
        for x in f:
            new_array = np.zeros((13, int(self.channel_num/13)))

            for i, value in enumerate(x.split(' ')):
                if not i in self.id_to_time_channel.keys(): break

                time, channel_id = self.id_to_time_channel[i]
                new_array[int(time)-1, channel_id] = float(value)

            data_array.append(new_array)

        self.data = np.stack(data_array, axis=0)
        print(self.data.shape)

    def __init__(self, txt_file, label_file, transform=None, test=False):
        self.parse_label(label_file)
        if test:
            pass
        else:
            self.parse_data(txt_file)
            self.transform=transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        sample['input'] = self.data[idx, :]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_channel(self):
        return self.channel_num


class ControlDatasetNoTimeAxis(Dataset):
    """Control dataset."""
    def gerControlChannels(self, label_file):
        self.channel = 0
        f = open(label_file, "r")
        for x in f:

            if ('Player' in x) or ('Gating' in x) or ('Bone' in x):
                break

            self.channel += 1


    def parse_data(self, txt_file, label_file, output_file):
        if not label_file is None:
            self.gerControlChannels(label_file)
        f = open(txt_file, "r")
        data_array = []
        for x in f:
            if self.channel is None:
                self.channel = len(x.split(' '))
            new_array = np.zeros((1, self.channel))

            x_split = x.split(' ')
            # for i, value in enumerate(x.split(' ')):
            #     new_array[i, 0] = float(value)
            for i in range(self.channel):
                new_array[0, i] = float(x_split[i])

            data_array.append(new_array)
        self.data = np.stack(data_array, axis=0)

        if not output_file is None:
            f = open(output_file, "r")
            data_array = []
            for x in f:
                if self.channel_output is None:
                    self.channel_output = len(x.split(' '))
                new_array = np.zeros((1, self.channel_output))

                x_split = x.split(' ')
                # for i, value in enumerate(x.split(' ')):
                #     new_array[i, 0] = float(value)
                for i in range(self.channel_output):
                    new_array[0, i] = float(x_split[i])

                data_array.append(new_array)
            self.data_output = np.stack(data_array, axis=0)

    def __init__(self, txt_file, transform=None, label_file=None, output_file=None):
        self.channel=None
        self.channel_output=None
        self.parse_data(txt_file, label_file, output_file)
        self.transform=transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        sample['input'] = self.data[idx, :]
        if not self.channel_output is None:
            sample['output'] = self.data_output[idx, :]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_channel(self):
        return self.channel

    def get_output_channel(self):
        return self.channel_output


class ControlDatasetArbitraryChannels(Dataset):
    def parse_data(self):
        f = open(self.input_file, "r")
        data_array = []
        for x in f:
            if self.training_config['input'][1] is None:
                self.training_config['input'][1] = [i for i in range(len(x.split(' ')))]
            new_array = np.zeros((1, len(self.training_config['input'][1])))

            x_split = x.split(' ')
            for i, j in enumerate(self.training_config['input'][1]):
                new_array[0, i] = float(x_split[j])

            data_array.append(new_array)
        self.data = np.stack(data_array, axis=0)

        f = open(self.output_file, "r")
        data_array = []
        for x in f:
            if self.training_config['output'][1] is None:
                self.training_config['output'][1] = [i for i in range(len(x.split(' ')))]
            new_array = np.zeros((1, len(self.training_config['output'][1])))

            x_split = x.split(' ')
            for i, j in enumerate(self.training_config['output'][1]):
                new_array[0, i] = float(x_split[j])

            data_array.append(new_array)
        self.data_output = np.stack(data_array, axis=0)
        assert(len(self.data) == len(self.data_output))

    def __init__(self, input_file, output_file, training_config, transform=None):
        self.input_file = input_file
        self.output_file = output_file
        self.training_config = training_config
        self.transform = transform
        self.parse_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        sample['input'] = self.data[idx, :]
        sample['output'] = self.data_output[idx, :]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_channel(self):
        return len(self.training_config['input'][1])

    def get_output_channel(self):
        return len(self.training_config['output'][1])

# Unit testing:
if __name__ == '__main__':
    dataset = ControlDataset('../dataset/Input.txt', '../dataset/InputLabels.txt',
                             transform=transforms.Compose([
                                get_smooth(list=('input',)), ]))
                                #ToTensor(list=('input', 'input_smooth'))]))

    print(dataset[10]['input'].shape)
    print(dataset[10]['input_smooth'].shape)

    print(np.average(dataset[10]['input'], axis=0))
    print(np.average(dataset[10]['input_smooth'], axis=0))

