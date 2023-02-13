import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import Library.Utility as utility

class Model(nn.Module):
    def __init__(self, input_channels, embedding_channels, time_range, window):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.embedding_channels = embedding_channels
        self.time_range = time_range
        self.window = window

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(time_range)[1:] * time_range / self.window, requires_grad=False) #Remove DC frequency

        intermediate_channels = int(input_channels/3)
        
        self.conv1 = nn.Conv1d(input_channels, intermediate_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.norm1 = utility.LN_v2(time_range)
        self.conv2 = nn.Conv1d(intermediate_channels, embedding_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.fc = torch.nn.ModuleList()
        for i in range(embedding_channels):
            self.fc.append(nn.Linear(time_range, 2))

        self.deconv1 = nn.Conv1d(embedding_channels, intermediate_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.denorm1 = utility.LN_v2(time_range)
        self.deconv2 = nn.Conv1d(intermediate_channels, input_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

    #Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        #Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        #Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        #Offset
        offset = rfft.real[:,:,0] / self.time_range #DC component

        return freq, amp, offset

    def forward(self, x):
        y = x

        #Signal Embedding
        y = y.reshape(y.shape[0], self.input_channels, self.time_range)

        y = self.conv1(y)
        y = self.norm1(y)
        y = F.elu(y)

        y = self.conv2(y)

        latent = y #Save latent for returning

        #Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        #Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi

        #Parameters    
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b] #Save parameters for returning

        #Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b

        signal = y #Save signal for returning

        #Signal Reconstruction
        y = self.deconv1(y)
        y = self.denorm1(y)
        y = F.elu(y)

        y = self.deconv2(y)

        y = y.reshape(y.shape[0], self.input_channels*self.time_range)

        return y, latent, signal, params





# import sys
# sys.path.append("../../../DeepLearning")

# import numpy as np
# import torch
# from torch.nn.parameter import Parameter
# import torch.nn as nn
# import torch.nn.functional as F
# import Library.Utility as utility

# class Model(nn.Module):
#     def __init__(self, input_channels, embedding_channels, time_range, window):
#         super(Model, self).__init__()
#         self.input_channels = input_channels
#         self.embedding_channels = embedding_channels
#         self.time_range = time_range
#         self.framerate = int((time_range - 1) / 2)

#         self.window = window

#         self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
#         self.args = Parameter(torch.from_numpy(np.linspace(-window/2, window/2, time_range, dtype=np.float32)), requires_grad=False)
#         self.freqs = Parameter(torch.fft.rfftfreq(time_range)[1:] * time_range / window, requires_grad=False) #Remove DC frequency

#         intermediate_channels = int(input_channels/3)

#         self.conv1 = nn.Conv1d(input_channels, intermediate_channels, time_range, stride=1, padding=self.framerate, dilation=1, groups=1, bias=True, padding_mode='zeros')
#         self.norm1 = utility.LN_v2(time_range)
#         self.conv2 = nn.Conv1d(intermediate_channels, embedding_channels, time_range, stride=1, padding=self.framerate, dilation=1, groups=1, bias=True, padding_mode='zeros')

#         self.fc = torch.nn.ModuleList()
#         for i in range(embedding_channels):
#             self.fc.append(nn.Linear(time_range, 2)) 

#         self.compressor = nn.Conv1d(self.framerate*embedding_channels, embedding_channels, time_range, stride=1, padding=self.framerate, dilation=1, groups=1, bias=True, padding_mode='zeros')

#         self.deconv1 = nn.Conv1d(embedding_channels, intermediate_channels, time_range, stride=1, padding=self.framerate, dilation=1, groups=1, bias=True, padding_mode='zeros')
#         self.denorm1 = utility.LN_v2(time_range)
#         self.deconv2 = nn.Conv1d(intermediate_channels, input_channels, time_range, stride=1, padding=self.framerate, dilation=1, groups=1, bias=True, padding_mode='zeros')

#         # self.deconv1 = nn.Conv1d(embedding_channels, intermediate_channels, time_range, stride=1, padding=self.framerate, dilation=1, groups=1, bias=True, padding_mode='zeros')
#         # self.denorm1 = utility.LN_v2(self.time_range)
#         # self.deconv2 = nn.Conv1d(intermediate_channels, input_channels, time_range, stride=1, padding=self.framerate, dilation=1, groups=1, bias=True, padding_mode='zeros')

#     #Returns the frequency for a function over a time window in s
#     def FFT(self, function, dim):
#         rfft = torch.fft.rfft(function, dim=dim)
#         magnitudes = rfft.abs()
#         spectrum = magnitudes[:,:,1:] #Spectrum without DC component
#         power = spectrum**2

#         #Frequency
#         freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

#         #Amplitude
#         amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

#         #Offset
#         offset = rfft.real[:,:,0] / self.time_range #DC component

#         return freq, amp, offset, power

#     def forward(self, x):
#         y = x

#         #Signal Embedding
#         y = y.reshape(y.shape[0], self.input_channels, self.time_range)

#         y = self.conv1(y)
#         y = self.norm1(y)
#         y = F.elu(y)

#         y = self.conv2(y)

#         latent = y #Save latent for returning

#         #Frequency, Amplitude, Offset
#         f, a, b, power = self.FFT(y, dim=2)

#         #Phase
#         p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
#         for i in range(self.embedding_channels):
#             v = self.fc[i](y[:,i,:])
#             p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi

#         #Parameters
#         a = torch.ones(a.shape)
#         b = torch.zeros(b.shape)
        
#         p = p.unsqueeze(2)
#         f = f.unsqueeze(2)
#         a = a.unsqueeze(2)
#         b = b.unsqueeze(2)
#         params = [p, f, a, b] #Save parameters for returning

#         phase = p.repeat(1,1,self.framerate)
#         freqs = self.freqs.reshape(1,1,-1).repeat(phase.shape[0], phase.shape[1], 1)

#         phase = phase.flatten(start_dim=1).unsqueeze(-1)
#         freqs = freqs.flatten(start_dim=1).unsqueeze(-1)
#         power = power.flatten(start_dim=1).unsqueeze(-1)

#         #Latent Reconstruction
#         y = power * torch.sin(self.tpi * (freqs * self.args + phase))

#         signal = y #Save signal for returning

#         # y = y.reshape(-1, self.embedding_channels, self.framerate, self.time_range)
#         # y = torch.sum(y, dim=2) / self.framerate

#         y = self.compressor(y)

#         #Signal Reconstruction
#         y = self.deconv1(y)
#         y = self.denorm1(y)
#         y = F.elu(y)

#         y = self.deconv2(y)

#         y = y.reshape(y.shape[0], self.input_channels*self.time_range)

#         return y, latent, signal, params