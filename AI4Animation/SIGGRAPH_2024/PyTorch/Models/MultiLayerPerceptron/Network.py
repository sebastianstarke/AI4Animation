import sys
sys.path.append("../../../PyTorch")

import Library.Utility as utility
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, rng, layers, activations, dropout, input_norm, output_norm):
        super(Model, self).__init__()

        self.rng = rng
        self.layers = layers
        self.activations = activations
        self.dropout = dropout
        self.Xnorm = Parameter(torch.from_numpy(input_norm), requires_grad=False)
        self.Ynorm = Parameter(torch.from_numpy(output_norm), requires_grad=False)
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        for i in range(len(layers)-1):
            self.W.append(self.weights([self.layers[i], self.layers[i+1]]))
            self.b.append(self.bias([1, self.layers[i+1]]))

    def weights(self, shape):
        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha = np.asarray(self.rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape), dtype=np.float32)
        return Parameter(torch.from_numpy(alpha), requires_grad=True)

    def bias(self, shape):
        return Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        x = utility.Normalize(x, self.Xnorm)
        y = x
        for i in range(len(self.layers)-1):
            y = F.dropout(y, self.dropout, training=self.training)
            y = y.matmul(self.W[i]) + self.b[i]
            if self.activations[i] != None:
                y = self.activations[i](y)
        return utility.Renormalize(y, self.Ynorm)

if __name__ == '__main__':
    name = "TrackerBody"
    # name = "FutureBody"
    # name = "TrackedUpperBody"
    # name = "UntrackedUpperBody"
    
    directory = "../../Datasets/"+name
    id = name + "_" + utility.GetFileID(__file__)
    load = directory
    save = directory+"/Training_"+id
    utility.MakeDirectory(save)

    InputName = "Input"
    OutputName = "Output"
    InputFile = load + "/" + InputName + ".bin"
    OutputFile = load + "/" + OutputName + ".bin"
    Xshape = utility.LoadTxtAsInt(load + "/" + InputName + "Shape.txt", True)
    Yshape = utility.LoadTxtAsInt(load + "/" + OutputName + "Shape.txt", True)
    Xnorm = utility.LoadTxt(load + "/" + InputName + "Normalization.txt", True)
    Ynorm = utility.LoadTxt(load + "/" + OutputName + "Normalization.txt", True)

    seed = 23456
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    
    epochs = 150
    batch_size = 32
    dropout = 0.25

    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    sample_count = Xshape[0]
    input_dim = Xshape[1]
    output_dim = Yshape[1]

    hidden_dim = 512

    layers = [input_dim, hidden_dim, hidden_dim, output_dim]
    activations = [F.elu, F.elu, None]

    print("Network Structure:", layers)

    network = Model(
        rng=rng,
        layers=layers,
        activations=activations,
        dropout=dropout,
        input_norm=Xnorm,
        output_norm=Ynorm
    )
    if torch.cuda.is_available():
        print('GPU found, training on GPU...')
        network = network.cuda()
    else:
        print('No GPU found, training on CPU...')
        
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

            yPred = network(xBatch)

            loss = loss_function(utility.Normalize(yPred, network.Ynorm), utility.Normalize(yBatch, network.Ynorm))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            error += loss.item()

        utility.SaveONNX(
            path=save + '/' + id + '_' + str(epoch+1) + '.onnx',
            model=network,
            input_size=(torch.zeros(1, input_dim)),
            input_names=['X'],
            output_names=['Y']
        )
        print('Epoch', epoch+1, error/(sample_count/batch_size))