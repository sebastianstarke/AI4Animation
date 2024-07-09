import sys
sys.path.append("../../../PyTorch")

import Library.Utility as utility
import Library.Plotting as plotting
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler
import Library.Modules as modules

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, encoder, estimator, decoder, xNorm, yNorm, codebook_channels, codebook_dim):
        super(Model, self).__init__()

        self.Encoder = encoder
        self.Estimator = estimator
        self.Decoder = decoder

        self.XNorm = xNorm
        self.YNorm = yNorm

        self.C = codebook_channels
        self.D = codebook_dim

    def sample_gumbel(self, tensor, scale, eps=1e-20):
        scale = scale.reshape(-1,1,1,1) #This is noise scale between 0 and 1
        noise = torch.rand_like(tensor) - 0.5 #This is random noise between -0.5 and 0.5
        samples = scale * noise + 0.5 #This is noise rescaled between 0 and 1 where 0.5 is default for 0 noise
        return -torch.log(-torch.log(samples + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature, scale):
        y = logits + self.sample_gumbel(logits, scale)
        return F.softmax(y / temperature, dim=-1)
    
    def gumbel_softmax(self, logits, temperature, scale):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, scale)

        y_soft = y.view(logits.shape)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        y_hard = y_hard.view(logits.shape)

        return y_soft, y_hard

    def sample(self, z, knn):
        z = z.reshape(-1, self.C, self.D)
        z = z.unsqueeze(0).repeat(knn.size(0), 1, 1, 1)
        z_soft, z_hard = self.gumbel_softmax(z, 1.0, knn)
        z_soft = z_soft.reshape(-1, self.C*self.D)
        z_hard = z_hard.reshape(-1, self.C*self.D)
        return z_soft, z_hard
    
    def forward(self, x, knn, t=None): #x=input, knn=samples, t=output
        #training
        if t is not None:
            #Normalize
            x = utility.Normalize(x, self.XNorm)
            t = utility.Normalize(t, self.YNorm)

            #Encode Y
            target_logits = self.Encoder(torch.cat((t,x), dim=1))
            target_probs, target = self.sample(target_logits, knn)

            #Encode X
            estimate_logits = self.Estimator(x)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            #Decode
            y = self.Decoder(target)

            #Renormalize
            return utility.Renormalize(y, self.YNorm), target_logits, target_probs, target, estimate_logits, estimate_probs, estimate
                
        #inference
        else:
            #Normalize
            x = utility.Normalize(x, self.XNorm)
            
            #Encode X
            estimate_logits = self.Estimator(x)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            #Decode
            y = self.Decoder(estimate)

            #Renormalize
            return utility.Renormalize(y, self.YNorm), estimate

if __name__ == '__main__':
    name = "LowerBody"
    directory = "../../Datasets/"+name
    id = name + "_" + utility.GetFileID(__file__)
    load = directory
    save = directory+"/Training_"+id
    utility.MakeDirectory(save)

    XFile = load + "/Input.bin"
    YFile = load + "/Output.bin"
    XShape = utility.LoadTxtAsInt(load + "/InputShape.txt", True)
    YShape = utility.LoadTxtAsInt(load + "/OutputShape.txt", True)
    Xlabels = load + "/InputLabels.txt"
    Ylabels = load + "/OutputLabels.txt"

    sample_count = XShape[0]
    input_dim = XShape[1]
    output_dim = YShape[1]

    utility.SetSeed(23456)

    epochs = 150
    batch_size = 32
    dropout = 0.25

    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    encoder_dim = 1024
    estimator_dim = 1024
    decoder_dim = 1024

    codebook_channels = 128
    codebook_dim = 8
    codebook_size = codebook_channels * codebook_dim
    
    print("Input Features:", input_dim)
    print("Output Features:", output_dim)

    network = utility.ToDevice(Model(
        encoder=modules.LinearEncoder(input_dim + output_dim, encoder_dim, encoder_dim, codebook_size, dropout),

        estimator=modules.LinearEncoder(input_dim, estimator_dim, estimator_dim, codebook_size, dropout),

        decoder=modules.LinearEncoder(codebook_size, decoder_dim, decoder_dim, output_dim, 0.0),

        xNorm=Parameter(torch.from_numpy(utility.LoadTxt(load + "/InputNormalization.txt", True)), requires_grad=False),
        yNorm=Parameter(torch.from_numpy(utility.LoadTxt(load + "/OutputNormalization.txt", True)), requires_grad=False),

        codebook_channels=codebook_channels,
        codebook_dim=codebook_dim
    ))
        
    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count, restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True)
    loss_function = nn.MSELoss()

    #Setup Plotting
    plt.ion()
    _, ax_latent = plt.subplots(1,5, figsize=(10,2))
    loss_history = utility.PlottingWindow("Loss History", ax=plt.subplots(figsize=(10,5)), drawInterval=500, yScale='log')
    def Item(value):
        return value.detach().cpu()

    #Start Generate Test Sequences
    Sequences = utility.LoadTxtRaw(load + "/Sequences.txt", False)
    Sequences = np.array(utility.Transpose2DList(Sequences)[0], dtype=np.int64)
    test_sequence_length = 60
    test_sequences = []
    for i in range(int(Sequences[-1])):
        indices = np.where(Sequences == (i+1))[0]
        intervals = int(np.floor(len(indices) / test_sequence_length))
        if intervals > 0:
            slices = np.array_split(indices, intervals)
            test_sequences += slices
    print("Test Sequences:",len(test_sequences))

    #Training Loop
    I = np.arange(sample_count)
    for epoch in range(epochs):
        scheduler.step()
        np.random.shuffle(I)
        error = 0.0
        for i in range(0, sample_count, batch_size):
            print('Progress', round(100 * i / sample_count, 2), "%", end="\r")
            train_indices = I[i:i+batch_size]

            xBatch = utility.ReadBatchFromFile(XFile, train_indices, XShape[1])
            yBatch = utility.ReadBatchFromFile(YFile, train_indices, YShape[1])

            prediction, target_logits, target_probs, target, estimate_logits, estimate_probs, estimate = network(
                xBatch, 
                knn=torch.ones(1, device=xBatch.device), 
                t=yBatch
            )

            mse_loss = loss_function(utility.Normalize(yBatch, network.YNorm), utility.Normalize(prediction, network.YNorm))
            matching_loss = loss_function(target, estimate)

            loss = mse_loss + matching_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            error += loss.item()

            loss_history.Add(
                (Item(mse_loss).item(), "MSE Loss"),
                (Item(matching_loss).item(), "Matching Loss")
            )

            if loss_history.Counter == 0:
                network.eval()

                idx = random.choice(test_sequences)
                xBatch = utility.ReadBatchFromFile(XFile, idx, XShape[1])
                yBatch = utility.ReadBatchFromFile(YFile, idx, YShape[1])

                input_sequences = []
                output_sequences = []
                target_sequences = []
                estimate_sequences = []
                predictions_sequences = []
                for s in range(100):
                    idx = random.choice(test_sequences)
                    xBatch = utility.ReadBatchFromFile(XFile, idx, XShape[1])
                    yBatch = utility.ReadBatchFromFile(YFile, idx, YShape[1])
                    prediction, target_logits, target_probs, target, estimate_logits, estimate_probs, estimate = network(xBatch, knn=torch.zeros(1, device=xBatch.device), t=yBatch)
                    input_sequences.append(Item(xBatch))
                    output_sequences.append(Item(yBatch))
                    target_sequences.append(Item(target))
                    estimate_sequences.append(Item(estimate))
                    predictions_sequences.append(Item(prediction))
                plotting.PCA2DSequence(ax_latent[0], test_sequence_length, input_dim, input_sequences, "Input")
                plotting.PCA2DSequence(ax_latent[1], test_sequence_length, output_dim, output_sequences, "Output")
                plotting.PCA2DSequence(ax_latent[2], test_sequence_length, codebook_size, target_sequences, "Target")
                plotting.PCA2DSequence(ax_latent[3], test_sequence_length, codebook_size, estimate_sequences, "Estimate")
                plotting.PCA2DSequence(ax_latent[4], test_sequence_length, output_dim, predictions_sequences, "Prediction")

                network.train()

                plt.gcf().canvas.draw()
                plt.gcf().canvas.start_event_loop(1e-1)

        print('Epoch', epoch+1, error/(sample_count/batch_size))
        loss_history.Print()

        utility.SaveONNX(
            path=save + '/' + id + '_' + str(epoch+1) + '.onnx',
            model=network,
            input_size=(torch.zeros(1, input_dim), torch.ones(1)),
            input_names=['X', 'K'],
            output_names=['Y', 'Code'],
            dynamic_axes={
                'K': {0: 'Size'}
            }
        )
        # torch.save(network, save + '/' + id + '_' + str(epoch+1) + '.pt')