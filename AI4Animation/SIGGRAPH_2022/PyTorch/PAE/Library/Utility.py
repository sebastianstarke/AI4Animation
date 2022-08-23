import sys
import os.path
import numpy as np
import array
import torch
import time
import random
import matplotlib.pyplot as plt

VERBOSE_STEP = 1000

class PlottingWindow():
    def __init__(self, title, ax=None, min=None, max=None, cumulativeHorizon=100, drawInterval=100):
        plt.ion()
        _, self.ax = plt.subplots() if ax is None else ax
        self.Title = title
        self.CumulativeHorizon = cumulativeHorizon
        self.DrawInterval = drawInterval
        self.YMin = min
        self.YMax = max
        self.YRange = [sys.float_info.max if min==None else min, sys.float_info.min if max==None else max]
        self.Functions = {} #string->[History, Horizon]
        self.Counter = 0

    def Add(self, *args): #arg->(value, label)
        for arg in args:
            value = arg[0]
            label = arg[1]
            if label not in self.Functions:
                self.Functions[label] = ([],[])
            function = self.Functions[label]
            function[0].append(value)
            function[1].append(sum(function[0][-self.CumulativeHorizon:]) / len(function[0][-self.CumulativeHorizon:]))

            self.YRange[0] = min(self.YRange[0], value) if self.YMin==None else self.YRange[0]
            self.YRange[1] = max(self.YRange[1], value) if self.YMax==None else self.YRange[1]

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
        self.ax.set_ylim(self.YRange[0], self.YRange[1])
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

def SetSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def Timestamp():
    return time.time()

def ElapsedTime(timestamp, output=False, formatted=False):
    elapsed = 1000 * (time.time() - timestamp) #milliseconds
    if output:
        print("{0:.3f}".format(elapsed), 'ms')
    if formatted:
        return "{0:.3f}".format(elapsed) + ' ms'
    else:
        return elapsed

#binaryFile = .bin data matrix of shape samples x features
#sampleCount = number of samples in the file
#featureCount = number of features per sample
def ReadBinary(binaryFile, sampleCount, featureCount):
    bytesPerLine = featureCount*4
    data = []
    with open(binaryFile, "rb") as f:
        for i in np.arange(sampleCount):
            if i % max(int(sampleCount / VERBOSE_STEP),1) == 0:
                print('Reading binary ' + binaryFile + '...', round(100 * i / sampleCount, 2), "%", end="\r")
            f.seek(i*bytesPerLine)
            bytes = f.read(bytesPerLine)
            data.append(np.float32(array.array('f', bytes)))
    print('Reading binary ' + binaryFile + '...', 100, "%", end="\r")
    print("")
    return np.concatenate(data).reshape(sampleCount, -1)
    # Example:
    # samples = 100
    # features = 2904
    # batch = ReadAll("Input.bin", samples, features)

#binaryFile = .bin data matrix of shape samples x features
#sampleIndices = list of sample indices from 0
#featureCount = number of features per sample
def ReadBatch(binaryFile, sampleIndices, featureCount):
    bytesPerLine = featureCount*4
    data = []
    with open(binaryFile, "rb") as f:
        for i in sampleIndices:
            f.seek(i*bytesPerLine)
            bytes = f.read(bytesPerLine)
            data.append(np.float32(array.array('f', bytes)))
    return np.concatenate(data).reshape(len(sampleIndices), -1)
    # Example:
    # batchSize = 32
    # samples = 100
    # features = 2904
    # batch = ReadBatch("Input.bin", np.random.randint(samples, size=batchSize), features)

def ToDevice(x):
    return x.cuda() if torch.cuda.is_available() else x

def FromDevice(x):
    return x.cpu() if torch.cuda.is_available() else x

def ToNumpy(X):
    return X.data.cpu().numpy()

def Transpose2DList(values):
    return [list(i) for i in zip(*values)]

def LoadTxtAsInt(path, debug=False):
    txt = np.loadtxt(path, dtype=np.int64)
    if(debug):
        print("Loaded " + path)
    return txt

def LoadTxt(path, debug=False):
    txt = np.float32(np.loadtxt(path))
    if(debug):
        print("Loaded " + path)
    return txt

def LoadTxtRaw(path, debug=False):
    with open(path) as f:
        content = f.read()
        content1D = content.split("\n")
        content2D = [entry.split(" ") for entry in content1D]
    if(debug):
        print("Loaded "+ path)
    return content2D