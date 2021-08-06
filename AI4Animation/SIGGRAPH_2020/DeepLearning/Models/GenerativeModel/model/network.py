'''
Author : Yiwei Zhao
Date : June 30, 2019

'''

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.distributions as tdist
import numpy.random as rd


class ControlNet(nn.Module):
    def __init__(self, n_channels=9, hidden_channels=(128, 256, 128), time_range=13):
        o_channels = time_range * n_channels
        super(ControlNet, self).__init__()
        self.ln1 = nn.Linear(n_channels*time_range, hidden_channels[0])
        self.relu1 = nn.ReLU()

        self.ln2 = nn.Linear(hidden_channels[0], hidden_channels[1])
        self.relu2 = nn.ReLU()

        self.ln3 = nn.Linear(hidden_channels[1], hidden_channels[2])
        self.relu3 = nn.ReLU()

        self.ln4 = nn.Linear(hidden_channels[2], o_channels)

    def forward(self, x, device, average=False):
        x = self.ln1(x)
        x = self.relu1(x)

        x = self.ln2(x)
        x = self.relu2(x)

        x = self.ln3(x)
        x = self.relu3(x)

        y = self.ln4(x)
        return y


class ControlNetAutoEncoder(nn.Module):
    def __init__(self, n_channels=9, hidden_channels=(256, 256), time_range=13, dropout=0.3, output_channel=None,
                 non_linear = "relu"):
        o_channels = time_range * n_channels
        super(ControlNetAutoEncoder, self).__init__()
        self.ln1 = nn.Linear(n_channels*time_range, hidden_channels[0])
        if non_linear == "relu":
            self.nl1 = nn.ReLU()
        elif non_linear == "elu":
            self.nl1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.Linear(hidden_channels[0], int(n_channels * time_range/2))
        if non_linear == "relu":
            self.nl2 = nn.ReLU()
        elif non_linear == "elu":
            self.nl2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.ln3 = nn.Linear(int(n_channels * time_range/2), hidden_channels[1])
        if non_linear == "relu":
            self.nl3 = nn.ReLU()
        elif non_linear == "elu":
            self.nl3 = nn.ELU()

        if output_channel is None:
            output_channel = o_channels
        self.ln4 = nn.Linear(hidden_channels[1], output_channel)

    def forward(self, x, device, noise=None, latent=False, std=None, factor=1.0):
        x = self.ln1(x)
        x = self.nl1(x)

        x = self.dropout1(x)

        x = self.ln2(x)
        x = self.nl2(x)
        h = self.dropout2(x)

        if (noise is None) or (noise==-1):
            # noise = torch.rand(x.shape).to(device)
            x = h
        else:
            torch.manual_seed(noise)
            # noise = torch.rand(h.shape).to(device)
            # std = torch.from_numpy(std)
            noise = torch.from_numpy(rd.normal(loc=0, scale=std)).to(torch.float32)
            x = h + noise * factor
            # x = h

        x = self.ln3(x)
        x = self.nl3(x)

        y = self.ln4(x)
        if latent:
            return y, h.detach().cpu().numpy()
        else:
            return y


class ControlNetVAE(nn.Module):
    def __init__(self, n_channels=9, hidden_channels=(256, 256), time_range=13, dropout=0.3, output_channel=None):
        o_channels = time_range * n_channels
        super(ControlNetVAE, self).__init__()
        self.ln1 = nn.Linear(n_channels*time_range, hidden_channels[0])
        self.relu1 = nn.ReLU()

        self.ln21 = nn.Linear(hidden_channels[0], int(n_channels * time_range/2))
        self.ln22 = nn.Linear(hidden_channels[0], int(n_channels * time_range / 2))

        self.ln3 = nn.Linear(int(n_channels * time_range/2), hidden_channels[1])
        self.relu3 = nn.ReLU()

        if output_channel is None:
            output_channel = o_channels
        self.ln4 = nn.Linear(hidden_channels[1], output_channel)

    def encode(self, x):
        h1 = self.relu1(self.ln1(x))
        return self.ln21(h1), self.ln22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu3(self.ln3(z))
        return self.ln4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

