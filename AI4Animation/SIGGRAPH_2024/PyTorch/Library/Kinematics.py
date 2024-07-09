# import sys
# import os.path
# import numpy as np
# import array
import torch
# import torch.nn as nn
# import time
# import random
# import matplotlib.pyplot as plt
# import warnings
# from collections import OrderedDict

#values is a 3D-tensor of Tx,Tz,Ry with shape [batch,values] where Ry is in angles
def CreateRoot(values):
    M = torch.zeros(values.shape[0], 4, 4, device=values.device)

    tx = values[:,0]
    ry = values[:,1]
    tz = values[:,2]

    ry = torch.deg2rad(ry)

    tensor_0 = torch.zeros_like(ry)
    tensor_1 = torch.ones_like(ry)
    R = torch.stack([
                    torch.stack([torch.cos(ry), tensor_0, torch.sin(ry)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-torch.sin(ry), tensor_0, torch.cos(ry)])
                    ]).permute(2,0,1)
    M[:,:3,:3] = R
    M[:,0,3] = tx
    M[:,1,3] = tensor_0
    M[:,2,3] = tz
    M[:,3,3] = tensor_1
    return M

#values is a 6D-tensor of Tx,Ty,Tz,Rx,Ry,Rz with shape [batch,values] where Rx,Ry,Rz are in angles
def CreateMatrix(values):
    M = torch.zeros(values.shape[0], 4, 4, device=values.device)

    x = values[:,3]
    y = values[:,4]
    z = values[:,5]

    x = torch.deg2rad(x)
    y = torch.deg2rad(y)
    z = torch.deg2rad(z)

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

#m is a tensor of shape Bx4x4
def GetPositions(m):
    return m[:, :, :3, 3]

#m is a tensor of shape Bx4x4
def GetForwards(m):
    return m[:, :, 2, :3]

#m is a tensor of shape Bx4x4
def GetUpwards(m):
    return m[:, :, 1, :3]

#m is a tensor of shape Bx4x4
def GetRights(m):
    return m[:, :, 0, :3]

#m are 4x4 matrices of size [batch,4,4], s are 4x4 matrices of size [batch,4,4]
def TransformationsFrom(m, s):
    return torch.bmm(s, m)

#m are 4x4 matrices of size [batch,4,4], s are 4x4 matrices of size [batch,4,4]
def TransformationsTo(values, space):
    space = torch.inverse(space)
    return torch.bmm(space, values)

#values=[batch,?,3], space=[batch,4,4], return=[batch,3]
def PositionsFrom(values, space):
    shape = values.shape

    values = values.reshape(values.shape[0], -1, 3)
    space = space.unsqueeze(1).repeat(1, values.shape[1], 1, 1)

    values = values.reshape(-1, values.shape[-1])
    space = space.reshape(-1, space.shape[2], space.shape[3])
    
    values = torch.add(space[:,:3,3], torch.matmul(space[:,:3,:3], values.unsqueeze(2)).squeeze(-1))
    
    values = values.reshape(shape)

    return values

#values=[batch,?,3], space=[batch,4,4], return=[batch,?,3]
def PositionsTo(values, space):
    space = torch.inverse(space)

    shape = values.shape
    
    values = values.reshape(values.shape[0], -1, 3)
    space = space.unsqueeze(1).repeat(1, values.shape[1], 1, 1)

    values = values.reshape(-1, values.shape[-1])
    space = space.reshape(-1, space.shape[2], space.shape[3])
    
    values = torch.add(space[:,:3,3], torch.matmul(space[:,:3,:3], values.unsqueeze(2)).squeeze(-1))
    
    values = values.reshape(shape)

    return values

#values=[batch,?,3], space=[batch,4,4], return=[batch,?,3]
def DirectionsFrom(values, space):
    shape = values.shape

    values = values.reshape(values.shape[0], -1, 3)
    space = space.unsqueeze(1).repeat(1, values.shape[1], 1, 1)

    values = values.reshape(-1, values.shape[-1])
    space = space.reshape(-1, space.shape[2], space.shape[3])
    
    values = torch.matmul(space[:,:3,:3], values.unsqueeze(2)).squeeze(-1)
    
    values = values.reshape(shape)

    return values

#values=[batch,?,3], space=[batch,4,4], return=[batch,?,3]
def DirectionsTo(values, space):
    space = torch.inverse(space)

    shape = values.shape

    values = values.reshape(values.shape[0], -1, 3)
    space = space.unsqueeze(1).repeat(1, values.shape[1], 1, 1)

    values = values.reshape(-1, values.shape[-1])
    space = space.reshape(-1, space.shape[2], space.shape[3])
    
    values = torch.matmul(space[:,:3,:3], values.unsqueeze(2)).squeeze(-1)
    
    values = values.reshape(shape)

    return values