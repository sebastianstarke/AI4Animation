import sys
import os.path
import numpy as np
import array
import torch
import torch.nn as nn
import time
import random
import matplotlib.pyplot as plt
import warnings
from collections import OrderedDict

VERBOSE_STEP = 1000

#values is a 6D-tensor of Tx,Ty,Tz,Rx,Ry,Rz with shape [batch,values]
def CreateMatrix(values):
    M = torch.zeros(values.shape[0], 4, 4, device=values.device)

    x = values[:,3]
    y = values[:,4]
    z = values[:,5]
    # x = torch.deg2rad(x)
    # y = torch.deg2rad(y)
    # z = torch.deg2rad(z)
    tensor_0 = torch.zeros_like(y)
    tensor_1 = torch.ones_like(y)
    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, torch.cos(x), -torch.sin(x)]),
                    torch.stack([tensor_0, torch.sin(x), torch.cos(x)])
                    ]).permute(2,0,1)
    RY = torch.stack([
                    torch.stack([torch.cos(y), tensor_0, torch.sin(y)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-torch.sin(y), tensor_0, torch.cos(y)])
                    ]).permute(2,0,1)
    RZ = torch.stack([
                    torch.stack([torch.cos(z), -torch.sin(z), tensor_0]),
                    torch.stack([torch.sin(z), torch.cos(z), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])
                    ]).permute(2,0,1)
    R = torch.bmm(RY, torch.bmm(RX, RZ))
    M[:,:3,:3] = R
    M[:,:3,3] = values[:,:3]
    M[:,3,3] = tensor_1
    return M

def GetPositions(m):
    return m[:, :, :3, 3]

def GetForwards(m):
    return m[:, :, 2, :3]

def GetUpwards(m):
    return m[:, :, 1, :3]

def GetRights(m):
    return m[:, :, 0, :3]

#p are 3D positions of size [batch,3], s are 4x4 matrices of size [batch,4,4]
def PositionsFrom(p, s):
    return torch.add(s[:,:3,3].unsqueeze(2), torch.matmul(s[:,:3,:3], p.unsqueeze(2))).squeeze(-1)

#p are 3D positions of size [batch,3], s are 4x4 matrices of size [batch,4,4]
def PositionsTo(p, s):
    s = torch.inverse(s)
    return torch.add(s[:,:3,3].unsqueeze(2), torch.matmul(s[:,:3,:3], p.unsqueeze(2))).squeeze(-1)

#m are 4x4 matrices of size [batch,4,4], s are 4x4 matrices of size [batch,4,4]
def TransformationsTo(m, s):
    return torch.bmm(torch.inverse(s), m)

#m are 4x4 matrices of size [batch,4,4], s are 4x4 matrices of size [batch,4,4]
def TransformationsFrom(m, s):
    return torch.bmm(s, m)

#FK Layer
class FKLayer(torch.nn.Module):
    def __init__(self, hierarchy):
        super(FKLayer, self).__init__()

        self.hierarchy = hierarchy #list of lists in format parent -> childs
        self.pairs = []
        self.offsets = []
        for item in self.hierarchy: 
            pair = [int(item[0]), int(item[1])]
            self.pairs.append(pair)
            offset = torch.tensor([float(item[2]), float(item[3]), float(item[4])])
            self.offsets.append(offset)

    # params is a tensor of shape batch x bones x 6 containing px,py,pz,rx,ry,rz
    def forward(self, params):
        transformations = []
        for i in range(len(self.hierarchy)):
            transformations.append(CreateMatrix(params[:,i,:]))
            if i > 0:
                transformations[-1][:,3,0] = self.offsets[i][0]
                transformations[-1][:,3,1] = self.offsets[i][1]
                transformations[-1][:,3,2] = self.offsets[i][2]
                
        for pair in self.pairs:
            parent = int(pair[0])
            child = int(pair[1])
            if parent != -1:
                transformations[child] = torch.matmul(
                    transformations[parent],
                    transformations[child]
                )
        return torch.stack(transformations, 1)

# #FK Layer
# class FKLayer(torch.nn.Module):
#     def __init__(self, hierarchy):
#         super(FKLayer, self).__init__()

#         self.hierarchy = hierarchy #list of lists in format parent -> childs

#     # offsets is a tensor of shape batch x (bones-1) x 3 (constant offsets from parents to childs)
#     # origin is a tensor of shape batch x 1 x 6 (learnable transformation from root to hip)
#     # angles is a tensor of shape batch x (bones-1) x 3 (learnable angles from parents to childs)
#     def forward(self, params, lengths):
#         transformations = []
#         for i in range(len(self.hierarchy)):
#             if i==0:
#                 transformations.append(self.matrix(params.shape[0], params[:,0:3], params[:,3:6], device=params.device))
#             else:
#                 values = [:,]
#                 tx = torch.zeros([params.shape[0]], dtype=torch.float32, device=params.device)
#                 ty = torch.zeros([params.shape[0]], dtype=torch.float32, device=params.device)
#                 tz = lengths[:,i]
#                 t = torch.stack((tx, ty, tz), 1)
#                 r = params[:,6+(i-1)*3:6+i*3]
#                 transformations.append(self.matrix(params.shape[0], t, r, device=params.device))
#         for sequence in self.hierarchy:
#             bone = sequence[0]
#             for i in range(1,len(sequence)):
#                 child = sequence[i]
#                 transformations[child] = torch.matmul(
#                     transformations[bone],
#                     transformations[child]
#                 )
#         return torch.stack(transformations, 1)

class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN_v3(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        return y

class LN_v4(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class BinaryLoader():
    def __init__(self, file, maxStack=10):
        self.File = file

class PlottingWindow():
    def __init__(self, title, ax=None, min=None, max=None, cumulativeHorizon=100, drawInterval=100, yScale='linear'):
        plt.ion()
        warnings.filterwarnings("ignore", message="Attempt to set non-positive ylim on a log-scaled axis will be ignored.")
        _, self.ax = plt.subplots() if ax is None else ax
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

class RunningStats:
    def __init__(self, dims):
        self.dims = dims
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else np.zeros(self.dims)

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else np.ones(self.dims)

    def sigma(self):
        return np.sqrt(self.variance())

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

def GetFileID(file):
    return os.path.basename(os.path.dirname(file))+"_"+os.path.basename(file)

def CollectPointers(file, max=None):
    pointers = []
    with open(file) as f:
        pivot = 0
        while(f.readline()):
            pointers.append(pivot)
            pivot = f.tell()
            if len(pointers) % VERBOSE_STEP == 0:
                print('Collecting data pointers for ' + file + ' - ' + str(len(pointers)), end="\r")
            if max != None and len(pointers) == max:
                break
    print("")
    # print("Loaded " + file)
    return np.array(pointers)

def ReadChunk(file, pointers):
    data = []
    with open(file) as f:
        for i in pointers:
            f.seek(i)
            data.append(np.float32(np.fromstring(f.readline(), sep=' ')))
    return np.concatenate(data).reshape(len(pointers), -1)

def SaveTxtFromList(X, name):
    count = 0
    with open(name + ".txt", 'w') as f:
        for line in X:
            count += 1
            item = ''
            for entry in line:
                item += str(entry) + ' '
            if count == len(X):
                f.write(item[:-1])
            else:
                f.write(item[:-1] + '\n')
            

def SaveTxtFromNumpy(X, name):
    np.savetxt(name + ".txt", X)

def SaveBinary(X, name):
    X.tofile(name + ".bin")

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

# #binaryFile = .bin data matrix of shape samples x features
# #sampleIndices = list of sample indices from 0
# #featureCount = number of features per sample
# def ReadBatch(binaryFile, sampleIndices, featureCount):
#     bytesPerLine = featureCount*4
#     data = []
#     with open(binaryFile, "rb") as f:
#         for i in sampleIndices:
#             f.seek(i*bytesPerLine)
#             bytes = f.read(bytesPerLine)
#             data.append(np.float32(array.array('f', bytes)))
#     return np.concatenate(data).reshape(len(sampleIndices), -1)
#     # Example:
#     # batchSize = 32
#     # samples = 100
#     # features = 2904
#     # batch = ReadBatch("Input.bin", np.random.randint(samples, size=batchSize), features)

#binaryFile = .bin data matrix of shape samples x features
#sampleIndices = list of sample indices from 0
#featureCount = number of features per sample
def ReadBatchFromFile(binaryFile, sampleIndices, featureCount):
    bytesPerLine = featureCount*4
    data = np.empty([len(sampleIndices), featureCount], dtype=np.float32)
    with open(binaryFile, "rb") as f:
        for i in range(len(sampleIndices)):
            f.seek(sampleIndices[i]*bytesPerLine)
            bytes = f.read(bytesPerLine)
            data[i] = np.float32(array.array('f', bytes))
    return ToDevice(torch.from_numpy(data))
    # Example:
    # batchSize = 32
    # samples = 100
    # features = 2904
    # batch = ReadBatch("Input.bin", np.random.randint(samples, size=batchSize), features)

#matrix = numpy data matrix of shape samples x features
#indices = list of sample indices from 0
def ReadBatchFromMatrix(matrix, indices):
    return ToDevice(torch.from_numpy(matrix[indices]))

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
    
def LoadNormalization(path, dim, debug=False):
    print("Loading " + path)
    norm = np.fromfile(path, dtype=np.float32)
    return norm.reshape(2, dim)

def MakeDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)

def ToDevice(x):
    return x.cuda() if torch.cuda.is_available() else x

def FromDevice(x):
    return x.cpu() if torch.cuda.is_available() else x

def ToNumpy(X):
    return X.data.cpu().numpy()

def ToPyTorch(X):
    return ToDevice(torch.tensor(X, requires_grad=True))

def LoadAsListOfInts(path, debug=False):
    file = open(path, 'r')
    lines = file.readlines()
    result = []
    for line in lines:
        split = line.split()
        map_object = map(int, split)
        values = list(map_object)
        result.append(values)
    return result

def PrintProgress(pivot, total, resolution=1000):
    step = max(int(total / resolution),1)
    if pivot % step == 0:
        print('Progress', round(100 * pivot / total, 2), "%", end="\r")

def LoadSequences(path, debug=False, lineCount=None):
    print("Loading "+ path)
    data = []
    with open(path) as file:
        pivot = 0
        for line in file:
            pivot += 1
            if debug:
                PrintProgress(pivot, lineCount)
            entry = line.rstrip().split(' ')
            data.append(entry[0])
            if pivot==lineCount:
                break
    data = np.array(data, dtype=np.int64)
    return data

def LoadTxtRaw(path, debug=False, lineCount=None, axis=None):
    print("Loading "+ path)
    data = []
    with open(path) as file:
        pivot = 0
        for line in file:
            pivot += 1
            if debug:
                PrintProgress(pivot, lineCount)
            entry = line.rstrip().split(' ')
            if axis != None:
                entry = entry[axis]
            data.append(entry)
                
    return data

def Transpose2DList(values):
    return [list(i) for i in zip(*values)]

def LoadTxtAsInt(path, debug=False):
    print("Loading " + path)
    txt = np.loadtxt(path, dtype=np.int64)
    return txt

def LoadTxtAsFloat(path, debug=False):
    print("Loading " + path)
    txt = np.loadtxt(path, dtype=np.float32)
    return txt

def LoadTxt(path, debug=False):
    print("Loading " + path)
    txt = np.float32(np.loadtxt(path))
    return txt

# def RawNormalize(X, mean, std):
#     return (X - mean) / std

# def RawRenormalize(X, mean, std):
#     return (X * std) + mean

def Normalize(X, N):
    mean = N[0]
    std = N[1]
    return (X - mean) / std

def Renormalize(X, N):
    mean = N[0]
    std = N[1]
    return (X * std) + mean

def NormalizeBN(X, N):
    if N.weight != None and N.bias != None:
        return (X - N.running_mean) / torch.sqrt(N.running_var + N.eps) * N.weight + N.bias
    else:
        return (X - N.running_mean) / torch.sqrt(N.running_var + N.eps)

def RenormalizeBN(X, N):
    if N.weight != None and N.bias != None:
        return (X-N.bias)/N.weight*torch.sqrt(N.running_var + N.eps) + N.running_mean
    else:
        return X * torch.sqrt(N.running_var + N.eps) + N.running_mean

def GetStateDict(model, names):
    dict = OrderedDict()
    for param_tensor in model.state_dict():
        if param_tensor in names :
            dict[param_tensor] = model.state_dict()[param_tensor]
    return dict

def PrintStateDict(model):
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def PrintParameters(model, learnable=None): #None: All, True: Learnable Parameters, False: Non-Learnable Parameters
    for name, param in model.named_parameters():
        if learnable == None:
            print(name, param)
        if learnable == True and param.requires_grad:
            print(name, param)
        if learnable == False and not param.requires_grad:
            print(name, param)

def FreezeParameters(model, value, names=None):
    for name, param in model.named_parameters():
        if names is None or name in names:
            param.requires_grad = not value

def GetParameters(model, learnable=None): #None: All, True: Learnable Parameters, False: Non-Learnable Parameters
    params = []
    for name, param in model.named_parameters():
        if learnable == None:
            params.append((name, param))
        if learnable == True and param.requires_grad:
            params.append((name, param))
        if learnable == False and not param.requires_grad:
            params.append((name, param))
    return params

def CountParameters(model, learnable=True):
    if learnable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def atan2(y, x):
    # tpi = self.tpi
    tpi = 2*np.pi
    ans = torch.atan(y/x)
    ans = torch.where( (x<0) * (y>=0), ans+0.5*tpi, ans)
    ans = torch.where( (x<0) * (y<0), ans-0.5*tpi, ans)
    return ans

def Gaussian(N, std, sym=True):
    if N < 1:
        return np.array([])
    if N == 1:
        return np.ones(1, 'd')
    odd = N % 2
    if not sym and not odd:
        N = N + 1
    n = np.arange(0, N) - (N - 1.0) / 2.0
    sig2 = 2 * std * std
    w = np.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w

def GMSELoss():
    def fn(output, target):
        filter = ToDevice(torch.tensor(Gaussian(26*3*121, 26*3*121/6.0), requires_grad=True))
        filter = filter.unsqueeze(0)
        loss = torch.mean(filter * (output - target)**2)
        return loss
    return fn

def Rescale(value, valueMin, valueMax, resultMin, resultMax):
    if valueMax-valueMin != 0.0:
        return (value-valueMin)/(valueMax-valueMin)*(resultMax-resultMin) + resultMin
    else:
        return value

def RainbowColor(index, count):
    frequency = 5.0/count
    return (
        Rescale(np.sin(frequency*index + 0.0) * (127.0) + 128.0, 0.0, 255.0, 0.0, 1.0),
        Rescale(np.sin(frequency*index + 2.0) * (127.0) + 128.0, 0.0, 255.0, 0.0, 1.0),
        Rescale(np.sin(frequency*index + 4.0) * (127.0) + 128.0, 0.0, 255.0, 0.0, 1.0),
        1.0
    )

def GetLabelIndicesExclude(file, names):
    indices = []
    with open(file, "r") as f:
        i = 0
        for x in f:
            valid = True
            if names is not None:
                for n in names:
                    if n in x:
                        valid = False
            if valid:
                indices.append(i)                        
            i += 1
    return torch.tensor(indices)

def GetLabelIndicesContain(file, names):
    indices = []
    with open(file, "r") as f:
        i = 0
        for x in f:
            valid = False
            if names is not None:
                for n in names:
                    if n in x:
                        valid = True
            if valid:
                indices.append(i)
            i += 1
    return torch.tensor(indices)

# def Gaussian_Noise(x, std, dims=None):
#     if std==0:
#         return x
#     elif dims==None:
#         return x + np.random.normal(0.0, std, (x.shape[0], x.shape[1])).astype(np.float32)
#     else:
#         noise = np.zeros((x.shape[0], x.shape[1]),dtype=np.float32)
#         noise[:,dims] = np.random.normal(0.0, std, (x.shape[0], len(dims))).astype(np.float32)
#         return x + noise

# def ScaleLoss(sub, scale):
#     if scale:
#         scale = np.array(scale)
#         if(len(scale.shape)<2):
#             scale = np.expand_dims(scale, 0)
#             scale = tf.constant(scale, dtype=tf.float32)
#             return sub*scale
#     else:
#         return sub

# def CollectPointers(file, max=None):
#     pointers = []
#     with open(file) as f:
#         pivot = 0
#         while(f.readline()):
#             pointers.append(pivot)
#             pivot = f.tell()
#             if len(pointers) % VERBOSE_STEP == 0:
#                 print('Collecting data pointers for ' + file + ' - ' + str(len(pointers)), end="\r")
#             if max != None and len(pointers) == max:
#                 break
#     print("")
#     # print("Loaded " + file)
#     return np.array(pointers)

# def ReadChunk(file, pointers):
#     data = []
#     with open(file) as f:
#         for i in pointers:
#             f.seek(i)
#             data.append(np.float32(np.fromstring(f.readline(), sep=' ')))
#     return np.concatenate(data).reshape(len(pointers), -1)
