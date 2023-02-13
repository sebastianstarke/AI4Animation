import sys
sys.path.append("../")

import Library.Utility as utility
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, gating_indices, gating_input, gating_hidden, gating_output, main_indices, main_input, main_hidden, main_output, dropout, input_norm, output_norm):
        super(Model, self).__init__()

        if len(gating_indices) + len(main_indices) != len(input_norm[0]):
            print("Warning: Number of gating features (" + str(len(gating_indices)) + ") and main features (" + str(len(main_indices)) + ") are not the same as input features (" + str(len(input_norm[0])) + ").")

        self.gating_indices = gating_indices
        self.main_indices = main_indices

        self.G1 = nn.Linear(gating_input, gating_hidden)
        self.G2 = nn.Linear(gating_hidden, gating_hidden)
        self.G3 = nn.Linear(gating_hidden, gating_output)

        self.E1 = ExpertLinear(gating_output, main_input, main_hidden)
        self.E2 = ExpertLinear(gating_output, main_hidden, main_hidden)
        self.E3 = ExpertLinear(gating_output, main_hidden, main_output)

        self.dropout = dropout
        self.Xnorm = Parameter(torch.from_numpy(input_norm), requires_grad=False)
        self.Ynorm = Parameter(torch.from_numpy(output_norm), requires_grad=False)

    def forward(self, x):
        x = utility.Normalize(x, self.Xnorm)

        #Gating
        g = x[:, self.gating_indices]

        g = F.dropout(g, self.dropout, training=self.training)
        g = self.G1(g)
        g = F.elu(g)

        g = F.dropout(g, self.dropout, training=self.training)
        g = self.G2(g)
        g = F.elu(g)

        g = F.dropout(g, self.dropout, training=self.training)
        g = self.G3(g)

        w = F.softmax(g, dim=1)

        #Main
        m = x[:, self.main_indices]

        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E1(m, w)
        m = F.elu(m)

        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E2(m , w)
        m = F.elu(m)

        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E3(m, w)

        return utility.Renormalize(m, self.Ynorm), w

#Output-Blended MoE Layer
class ExpertLinear(torch.nn.Module):
    def __init__(self, experts, input_dim, output_dim):
        super(ExpertLinear, self).__init__()

        self.experts = experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = self.weights([experts, input_dim, output_dim])
        self.b = self.bias([experts, 1, output_dim])

    def forward(self, x, weights):
        y = torch.zeros((x.shape[0], self.output_dim), device=x.device, requires_grad=True)
        for i in range(self.experts):
            y = y + weights[:,i].unsqueeze(1) * (x.matmul(self.W[i,:,:]) + self.b[i,:,:])
        return y

    def weights(self, shape):
        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha = np.asarray(np.random.uniform(low=-alpha_bound, high=alpha_bound, size=shape), dtype=np.float32)
        return Parameter(torch.from_numpy(alpha), requires_grad=True)

    def bias(self, shape):
        return Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=True)

if __name__ == '__main__':

    load = "Data"
    gating_indices = torch.tensor([(429 + i) for i in range(130)])
    main_indices = torch.tensor([(0 + i) for i in range(429)])

    save = "./Training"

    InputFile = load + "/Input.bin"
    OutputFile = load + "/Output.bin"
    # InputFile = utility.ReadBinary(InputFile, sample_count, input_dim)
    # OutputFile = utility.ReadBinary(OutputFile, sample_count, output_dim)
    Xshape = utility.LoadTxtAsInt(load + "/InputShape.txt", True)
    Yshape = utility.LoadTxtAsInt(load + "/OutputShape.txt", True)
    Xnorm = utility.LoadTxt(load + "/InputNormalization.txt", True)
    Ynorm = utility.LoadTxt(load + "/OutputNormalization.txt", True)

    utility.SetSeed(23456)

    epochs = 150
    batch_size = 32
    dropout = 0.3
    gating_hidden = 64
    main_hidden = 1024
    experts = 8

    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    sample_count = Xshape[0]
    input_dim = Xshape[1]
    output_dim = Yshape[1]

    network = utility.ToDevice(Model(
        gating_indices=gating_indices, 
        gating_input=len(gating_indices), 
        gating_hidden=gating_hidden, 
        gating_output=experts, 
        main_indices=main_indices, 
        main_input=len(main_indices), 
        main_hidden=main_hidden, 
        main_output=output_dim,
        dropout=dropout,
        input_norm=Xnorm,
        output_norm=Ynorm
    ))

    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count, restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True)
    loss_function = torch.nn.MSELoss()

    I = np.arange(sample_count)
    for epoch in range(epochs):
        scheduler.step()
        np.random.shuffle(I)
        error = 0.0
        for i in range(0, sample_count, batch_size):
            print('Progress', round(100 * i / sample_count, 2), "%", end="\r")
            train_indices = I[i:i+batch_size]

            xBatch = utility.ReadBatchFromFile(InputFile, train_indices, input_dim)
            yBatch = utility.ReadBatchFromFile(OutputFile, train_indices, output_dim)
            # xBatch = utility.ReadBatchFromMatrix(InputFile, train_indices)
            # yBatch = utility.ReadBatchFromMatrix(OutputFile, train_indices)

            yPred, gPred = network(xBatch)

            loss = loss_function(utility.Normalize(yPred, network.Ynorm), utility.Normalize(yBatch, network.Ynorm))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            error += loss.item()
    
        utility.SaveONNX(
            path=save+'/'+str(epoch+1)+'.onnx',
            model=network,
            input_size=input_dim,
            input_names=['X'],
            output_names=['Y', 'W']
        )
        print('Epoch', epoch+1, error/(sample_count/batch_size))