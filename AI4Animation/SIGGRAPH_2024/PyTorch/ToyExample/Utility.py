import sys
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import warnings

class LinearEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, dropout):
        super(LinearEncoder, self).__init__()

        self.InputSize = input_size
        self.OutputSize = output_size

        self.Dropout = dropout

        self.L1 = nn.Linear(input_size, hidden1_size)
        self.L2 = nn.Linear(hidden1_size, hidden2_size)
        self.L3 = nn.Linear(hidden2_size, output_size)
        
    def forward(self, input):
        z = input

        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L1(z)
        z = F.elu(z)

        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L2(z)
        z = F.elu(z)

        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L3(z)

        return z

class PlottingWindow():
    def __init__(self, title, ax=None, min=None, max=None, cumulativeHorizon=100, drawInterval=100, yScale='linear'):
        plt.ion()
        warnings.filterwarnings("ignore", message="Attempt to set non-positive ylim on a log-scaled axis will be ignored.")
        if ax is None:
            _, self.ax = plt.subplots()
        else:
            self.ax = ax
        self.Title = title
        self.CumulativeHorizon = cumulativeHorizon
        self.DrawInterval = drawInterval
        self.YMin = min
        self.YMax = max
        self.YRange = [sys.float_info.max if min==None else min, sys.float_info.min if max==None else max]
        self.Functions = {} #string->[History, Horizon]
        self.Counter = 0
        self.YScale = yScale

    def Add(self, *args): #arg->(value, label)
        for arg in args:
            value = arg[0]
            label = arg[1]
            if label not in self.Functions:
                self.Functions[label] = ([],[]) #(Value, Cumulative)
            function = self.Functions[label]
            function[0].append(value)
            cumulative = sum(function[0][-self.CumulativeHorizon:]) / len(function[0][-self.CumulativeHorizon:])
            function[1].append(cumulative)
            self.YRange[0] = min(self.YRange[0], 0.5*cumulative) if self.YMin==None else self.YRange[0]
            self.YRange[1] = max(self.YRange[1], 2*cumulative) if self.YMax==None else self.YRange[1]

        self.Counter += 1
        if self.Counter >= self.DrawInterval:
            self.Counter = 0
            self.Draw()

    def Draw(self):
        self.ax.cla()
        self.ax.set_title(self.Title)
        for label in self.Functions.keys():
            function = self.Functions[label]
            step = max(int(len(function[0])/self.DrawInterval), 1)
            self.ax.plot(function[0][::step], label=label + " (" + str(round(self.CumulativeValue(label), 3)) + ")")
            self.ax.plot(function[1][::step], c=(0,0,0))
        self.ax.set_yscale(self.YScale)
        self.ax.set_ylim((self.YRange[0],self.YRange[1]))
        self.ax.legend()
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(1e-5)

    def Value(self, label=None):
        if label==None:
            return sum(x[0][-1] for x in self.Functions.values())
        else:
            return self.Functions[label][0][-1]

    def CumulativeValue(self, label=None):
        if label==None:
            return sum(x[1][-1] for x in self.Functions.values())
        else:
            return self.Functions[label][1][-1]
        
    def Print(self, digits=5):
        output = ""
        for name in self.Functions.keys():
            output = output + name + ": " + str(round(self.CumulativeValue(name), digits)) + " "
        print(output)

def SetSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def SaveONNX(path, model, input_size, input_names, output_names, dynamic_axes=None):
    FromDevice(model)
    torch.onnx.export(
        model,                            # model being run
        # (torch.zeros(1, input_size)),
        # (torch.zeros(1, input_size), torch.zeros(1)),          # model input (or a tuple for multiple inputs) -> torch.zeros(1, first_size), torch.zeros(1, second_size)),
        input_size,
        path,            # where to save the model (can be a file or file-like object)
        training=torch.onnx.TrainingMode.EVAL, #EVAL, PRESERVE, TRAINING
        export_params=True,                 # store the trained parameter weights inside the model file
        opset_version=12,                    # the ONNX version to export the model to
        do_constant_folding=False,          # whether to execute constant folding for optimization
        input_names = input_names,                # the model's input names
        output_names = output_names,                # the model's output names
        dynamic_axes = dynamic_axes
    )
    ToDevice(model)
    
def MakeDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)

def ToDevice(x):
    return x.cuda() if torch.cuda.is_available() else x

def FromDevice(x):
    return x.cpu() if torch.cuda.is_available() else x

def PrintProgress(pivot, total, resolution=1000):
    step = max(int(total / resolution),1)
    if pivot % step == 0:
        print('Progress', round(100 * pivot / total, 2), "%", end="\r")


def Normalize(X, N):
    mean = N[0]
    std = N[1]
    return (X - mean) / std

def Renormalize(X, N):
    mean = N[0]
    std = N[1]
    return (X * std) + mean