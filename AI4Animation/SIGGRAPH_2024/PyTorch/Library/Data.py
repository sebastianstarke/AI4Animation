import sys
sys.path.append("../../../PyTorch")

import Library.Utility as utility

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Binary():
    def __init__(self, directory, name):
        self.Name = name
        self.File = directory + "/" + name + ".bin"
        self.Labels = directory + "/" + name + "Labels.txt"
        self.Shape = utility.LoadTxtAsInt(directory + "/" + name + "Shape.txt", True)
        self.Norm = Parameter(torch.from_numpy(utility.LoadTxt(directory + "/" + name + "Normalization.txt", True)), requires_grad=False)
        self.Samples = self.Shape[0]
        self.Dimensions = self.Shape[1]
        print("Binary Directory:", directory, "File:", name)
        
        # print("Loaded:", self.File)
        # print("Loaded:", self.Shape)
        # print("Loaded:", self.Norm)
    
    def ReadBatch(self, indices): #Indices of shape [Batch]
        return utility.ReadBatchFromFile(self.File, indices, self.Dimensions)
    
    def ReadBatches(self, indices): #Indices of shape [Sequence, Batch]
        batches = utility.ReadBatchFromFile(self.File, indices.flatten(), self.Dimensions)
        batches = batches.reshape(indices.shape)
        return batches

class Sequences():
    def __init__(self, path):
        print("Generating Data Sequences")
        file = utility.LoadSequences(path)
        self.Pointers = []
        for i in range(file[-1]):
            utility.PrintProgress(i, file[-1])
            indices = np.where(file == (i+1))[0]
            self.Pointers.append([indices[0], indices[-1]])
            # for j in range(indices.shape[0]):
            #     slice = [indices[j], indices[0], indices[-1]]
            #     self.Pointers.append(slice)
        self.Pointers = np.array(self.Pointers)
        self.Size = len(self.Pointers)

    def SampleIndices(self, batch, length, startPadding=0, endPadding=0):
        ranges = random.choices(self.Pointers, k=batch)

        min = []
        max = []
        for i in range(batch):
            min.append(ranges[i][0]+startPadding)
            max.append(ranges[i][1]-endPadding-length)

        pivots = torch.from_numpy(np.random.randint(min, max, size=batch))
        pivots = pivots.reshape(-1, 1)

        span = torch.arange(length)
        span = span.reshape(1, -1)
        span = span.repeat(batch, 1)

        result = pivots + span
        result = result.swapaxes(0, 1)

        return result