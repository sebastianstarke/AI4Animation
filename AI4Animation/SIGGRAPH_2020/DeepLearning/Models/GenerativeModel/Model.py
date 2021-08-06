'''
Author : Yiwei Zhao
Date : July 3, 2019
'''

from optparse import OptionParser
import os, sys
import numpy as np

# Fix for cloud training system path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../'))
sys.path.append(dir_path)

import torch

from model.network import ControlNetAutoEncoder
import model.data_loader as DL

import json


def load_average_from_txt(name_list):
    average_list = {}
    std_list = {}
    for name in name_list.keys():
        '''f = open(name_list[name], "r")
        for i, x in enumerate(f):
            l = x.split()
            ave = np.array(len(l), )
            std = np.array(len(l), )
            for j in range(len(l)):
                ave[j] = float(l[j])'''
        load = np.float32(np.loadtxt(name_list[name][0]))
        cutoff = name_list[name][1]
        if cutoff is None:
            average_list[name] = load[0,:]
            std_list[name] = load[1,:]
        else:
            average_list[name] = load[0, :cutoff]
            std_list[name] = load[1, :cutoff]
        std_list[name][np.where(std_list[name] == 0)] = 1
    return average_list, std_list


def get_average(name_list, dataset_dir):
    average_list = {}
    std_list = {}
    for name in name_list:
        if os.path.exists(dataset_dir + 'average_' + name + '.npy') and os.path.exists(dataset_dir + 'std_' + name + '.npy'):
            average_list[name] = np.load(dataset_dir + 'average_' + name + '.npy')
            std_list[name] = np.load(dataset_dir + 'std_' + name + '.npy')
        else:
            raise Exception("average value not found")

    return average_list, std_list


class Model:
    def Setup(self, path):
        # Change the path to the checkpoint to inference different models:
        #self.model_path = "/".join(path.split("/")[:-1]) + "/"
        self.model_path = path + "/"


        # Don't need to change the following
        check_point = self.model_path + 'model_weight/CP_gen.pth'
        std_path = self.model_path + 'model_weight/CP_latent_std.npy'
        dataset_dir = self.model_path + 'channels_and_average/'
        configure_path = self.model_path + 'model_weight/configure.txt'

        with open(configure_path) as f:
            self.config = json.loads(f.read())

        self.n_channels = int(self.config["input_channel"])
        if self.config["output_channel"] is None:
            self.output_channel = self.n_channels
        else:
            self.output_channel = int(self.config["output_channel"])

        self.time_range = int(self.config["time_range"])
        self.control_only = self.config["control_only"]

        self.control_net = ControlNetAutoEncoder(n_channels=self.n_channels,
                                                 time_range=self.time_range,
                                                 dropout=0.0,
                                                 output_channel=self.output_channel)
        self.device = torch.device("cpu")
        self.control_net.load_state_dict(torch.load(check_point, map_location=lambda storage, loc: storage))
        self.control_net = self.control_net.to(self.device).eval()

        # average_list, std_list = get_average(['input', ], dataset_dir)
        self.average_list, self.std_list = load_average_from_txt({'input': (dataset_dir + 'InputNorm.txt', self.n_channels),
                                                        'output': (dataset_dir + 'OutputNorm.txt', self.output_channel),})

        self.average_list_input = {'input': self.average_list['input']}
        self.std_list_input = {'input': self.std_list['input']}
        self.latent_std = np.load(std_path)

        self.pre_funcs = [DL.sub_average(self.average_list_input, self.std_list_input),
                     DL.ToTensor(list=('input', ), batch=True)]

    def GetInputDim(self):
        return self.n_channels + 2

    def GetOutputDim(self):
        return self.output_channel

    def Predict(self, X_compose, plot=False):
        # X is a numpy array with shape [N]
        # Run preprocessing
        X = X_compose[:-2]
        noise_seed = X_compose[-2]
        factor = X_compose[-1]
        input = np.zeros((1, self.n_channels))
        for i in range(self.n_channels):
            input[0, i] = X[i]

        gt = input.copy()
        sample = {'input': input}
        for pre_func in self.pre_funcs:
            sample = pre_func(sample)

        inputs = sample['input'].type('torch.FloatTensor')
        inputs = inputs.to(self.device)
        ori_shape = inputs.shape
        inputs = inputs.view((-1, ori_shape[1] * ori_shape[2]))
        outputs = self.control_net(inputs, device='cpu', noise=int(noise_seed), std=self.latent_std, factor=float(factor))
        outputs = outputs.view((self.output_channel, self.time_range)).detach().numpy().transpose((1, 0))
        if self.control_only:
            outputs = np.multiply(outputs, self.std_list['input']) + self.average_list['input']
        else:
            outputs = np.multiply(outputs, self.std_list['output']) + self.average_list['output']

        '''inputs = inputs.view((ori_shape[1], ori_shape[2])).detach().numpy().transpose((1, 0))
        inputs = np.multiply(inputs, self.std_list['input']) + self.average_list['input']
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(range(outputs.shape[1]), outputs[0, :], label="predict")
            plt.plot(range(gt.shape[1]), gt[0, :], label="gt")
            plt.plot(range(inputs.shape[1]), inputs[0, :], label="inputs")
            name = None
            plt.title(name)
            plt.legend(loc='upper left')
            plt.show()'''

        return outputs.flatten()


if __name__ == '__main__':
    f = open('dataset/Input.txt', "r")
    model = Model()
    model.Setup('ball_control_checkpoint_new')
    for x in f:
        X = np.array([float(a) for a in x.split(' ')] + [10] + [1.0])
        # X = np.array([float(a) for a in x.split(' ')] + [666])
        # inference(X)
        print(np.mean(np.abs(X[:208] - model.Predict(X, plot=True))))
