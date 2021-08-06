'''
Author : Yiwei Zhao
Date : Oct 9, 2019

'''

import torch.nn.functional as F
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_channels=9, hidden_channels=(128, 64, 32), time_range=13):
        super(Discriminator, self).__init__()
        self.ln1 = nn.Linear(n_channels*time_range*2, hidden_channels[0])
        self.relu1 = nn.ReLU()

        self.ln2 = nn.Linear(hidden_channels[0], hidden_channels[1])
        self.relu2 = nn.ReLU()

        self.ln3 = nn.Linear(hidden_channels[1], hidden_channels[2])
        self.relu3 = nn.ReLU()

        self.ln4 = nn.Sequential(nn.Linear(hidden_channels[2], 1),
                                 nn.Sigmoid())

    def forward(self, condition, output, device, average=False):
        x = torch.cat((condition, output), 1)
        x = self.ln1(x)
        x = self.relu1(x)

        x = self.ln2(x)
        x = self.relu2(x)

        x = self.ln3(x)
        x = self.relu3(x)

        y = self.ln4(x)
        return y


class DiscriminatorConless(nn.Module):
    def __init__(self, n_channels=9, hidden_channels=(128, 64, 32), time_range=13, dropout=0.3,
                 non_linear="relu"):
        super(DiscriminatorConless, self).__init__()
        self.ln1 = nn.Linear(n_channels*time_range, hidden_channels[0])
        if non_linear == "relu":
            self.nl1 = nn.ReLU()
        elif non_linear == "elu":
            self.nl1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.Linear(hidden_channels[0], hidden_channels[1])
        if non_linear == "relu":
            self.nl2 = nn.ReLU()
        elif non_linear == "elu":
            self.nl2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.ln3 = nn.Linear(hidden_channels[1], hidden_channels[2])
        if non_linear == "relu":
            self.nl3 = nn.ReLU()
        elif non_linear == "elu":
            self.nl3 = nn.ELU()

        self.ln4 = nn.Sequential(nn.Linear(hidden_channels[2], 1),
                                 nn.Sigmoid())

    def forward(self, x, device, average=False):
        x = self.ln1(x)
        x = self.nl1(x)

        x = self.ln2(x)
        x = self.nl2(x)


        # x = self.dropout2(x)


        x = self.ln3(x)
        x = self.nl3(x)


        y = self.ln4(x)
        return y

