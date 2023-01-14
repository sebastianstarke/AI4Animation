import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_channels, embedding_channels, time_range, key_range, window):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.embedding_channels = embedding_channels
        self.time_range = time_range
        self.key_range = key_range

        self.window = window
        self.time_scale = key_range/time_range

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(time_range)[1:] * (time_range * self.time_scale) / self.window, requires_grad=False) #Remove DC frequency

        intermediate_channels = int(input_channels/3)
        
        self.conv1 = nn.Conv1d(input_channels, intermediate_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_conv1 = nn.BatchNorm1d(num_features=intermediate_channels)
        self.conv2 = nn.Conv1d(intermediate_channels, embedding_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_conv2 = nn.BatchNorm1d(num_features=embedding_channels)

        self.fc = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        for i in range(embedding_channels):
            self.fc.append(nn.Linear(time_range, 2))
            self.bn.append(nn.BatchNorm1d(num_features=2))

        self.deconv1 = nn.Conv1d(embedding_channels, intermediate_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_deconv1 = nn.BatchNorm1d(num_features=intermediate_channels)
        self.deconv2 = nn.Conv1d(intermediate_channels, input_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

    def inference(self, x):
        x = torch.from_numpy(x.copy().reshape(1, self.input_channels*self.time_range))
        _, _, _, params = self.forward(x)
        y = torch.cat((
            params[0].flatten(), #p
            params[1].flatten(), #f
            params[2].flatten(), #a
            params[3].flatten()  #b
        ), dim=0).detach().numpy()
        return y

    def input_dim(self):
        return self.input_channels * self.time_range

    def output_dim(self):
        return self.embedding_channels*4

    #Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        #Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)
        freq = freq / self.time_scale

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
        y = self.bn_conv1(y)
        y = F.elu(y)

        y = self.conv2(y)
        y = self.bn_conv2(y)
        y = F.elu(y)

        latent = y #Save latent for returning

        #Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        #Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            v = self.bn[i](v)
            p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi

        #Parameters
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b] #Save parameters for returning

        #Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b
        # y = torch.cat(
        #     (
        #     a * torch.sin(self.tpi * (f * self.args + p)) + b,
        #     a * torch.cos(self.tpi * (f * self.args + p)) + b
        #     ),
        #     dim=1
        # )

        signal = y #Save signal for returning

        #Signal Reconstruction
        y = self.deconv1(y)
        y = self.bn_deconv1(y)
        y = F.elu(y)

        y = self.deconv2(y)

        y = y.reshape(y.shape[0], self.input_channels*self.time_range)

        return y, latent, signal, params