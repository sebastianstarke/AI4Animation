import Utility as utility
import AdamWR.adamw as adamw
import AdamWR.cyclic_scheduler as cyclic_scheduler

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# import CVQ as cvq
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, encoder, estimator, quantizer, decoder, xNorm, yNorm, codebook_channels, codebook_dim):
        super(Model, self).__init__()

        self.Encoder = encoder
        self.Estimator = estimator
        self.Quantizer = quantizer
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

    def sample(self, z, k):
        z = z.reshape(-1, self.C, self.D)
        z = z.unsqueeze(0).repeat(k.size(0), 1, 1, 1)
        z_soft, z_hard = self.gumbel_softmax(z, 1.0, k)
        z_soft = z_soft.reshape(-1, self.C*self.D)
        z_hard = z_hard.reshape(-1, self.C*self.D)
        return z_soft, z_hard
    
    def forward(self, x, k, t=None): #x = input, k = k-th nearest neighbor, t = output
        #training
        if t is not None:
            #Normalize
            x = utility.Normalize(x, self.XNorm)
            t = utility.Normalize(t, self.YNorm)

            #Encode Y
            target_logits = self.Encoder(torch.cat((t,x), dim=-1))
            target_soft, target_hard = self.sample(target_logits, torch.ones(1, device=x.device))

            #Encode X
            estimate_logits = self.Estimator(x)
            estimate_soft, estimate_hard = self.sample(estimate_logits, torch.ones(1, device=x.device))

            #Calculate Loss
            mse_function = nn.MSELoss()
            matching_loss = mse_function(estimate_soft, target_soft)
            # matching_loss = mse_function(torch.pdist(target_soft.swapaxes(0,1), p=2), torch.pdist(estimate_soft.swapaxes(0,1), p=2))
            
            #Quantize
            code = target_soft
            if self.Quantizer != None:
                code, vq_loss = self.Quantizer(code)
            else:
                vq_loss = torch.tensor(0.0).to(code.device)

            #Decode
            y = self.Decoder(code)
            
            #Renormalize
            return utility.Renormalize(y, self.YNorm), matching_loss, vq_loss
        #inference
        else:
            #Normalize
            x = utility.Normalize(x, self.XNorm)

            #Encode X
            estimate_logits = self.Estimator(x)
            estimate_soft, estimate_hard = self.sample(estimate_logits, k)

            #Quantize
            code = estimate_hard
            if self.Quantizer != None:
                code, _ = self.Quantizer(code)

            #Decode
            y = self.Decoder(code)

            #Renormalize
            return utility.Renormalize(y, self.YNorm), code

if __name__ == '__main__':
    save = "Training"
    utility.MakeDirectory(save)
    utility.SetSeed(23456)

    pairs = 10000
    resolution = 100
    X = torch.rand(pairs).reshape(-1,1)
    Y = X * torch.pow(torch.linspace(-2, 2, steps=resolution), 2.0)
    X = torch.cat((X,X), dim=0)
    Y = torch.cat((Y,-Y), dim=0)

    XNorm = torch.cat((torch.mean(X, dim=0), torch.std(X, dim=0)), dim=0)
    YNorm = torch.cat((torch.mean(Y, dim=0), torch.std(Y, dim=0)), dim=0)

    XNorm[1] = torch.where(XNorm[1] < 1e-5, 1.0, XNorm[1])
    YNorm[1] = torch.where(YNorm[1] < 1e-5, 1.0, YNorm[1])

    sample_count = X.shape[0]
    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    epochs = 150
    batch_size = 32
    dropout = 0.25

    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    encoder_dim = 2048
    estimator_dim = 2048
    decoder_dim = 2048

    codebook_channels = 128
    codebook_dim = 8
    codebook_size = codebook_channels * codebook_dim
    
    print("Input Features:", input_dim)
    print("Output Features:", output_dim)
    # print("Codebook Channels:", codebook_channels)
    # print("Codebook Dimensions:", codebook_dim)
    # print("Codebook Size:", codebook_size)

    network = utility.ToDevice(Model(
        encoder=utility.LinearEncoder(input_dim + output_dim, encoder_dim, encoder_dim, codebook_size, 0.0),

        estimator=utility.LinearEncoder(input_dim, estimator_dim, estimator_dim, codebook_size, dropout),

        # quantizer=cvq.VectorQuantizer(embed_channels=codebook_channels, embed_dim=codebook_dim, beta=0.25, distance='l2', anchor='closest', first_batch=False, contras_loss=True),
        quantizer=None,

        decoder=utility.LinearEncoder(codebook_size, decoder_dim, decoder_dim, output_dim, 0.0),

        xNorm=Parameter(XNorm, requires_grad=False),
        yNorm=Parameter(YNorm, requires_grad=False),

        codebook_channels = codebook_channels,
        codebook_dim = codebook_dim
    ))
    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count, restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True)

    #Setup Plotting
    plt.ion()
    plt.figure(1, figsize=(10, 6))
    ax1 = plt.subplot(212)
    ax2 = plt.subplot(231)
    ax3 = plt.subplot(232)
    ax4 = plt.subplot(233)
    loss_history = utility.PlottingWindow("Loss History", ax=ax1, drawInterval=100, yScale='log')

    def Item(value):
        return value.detach().cpu()

    #Training Loop
    I = np.arange(sample_count)
    for epoch in range(epochs):
        scheduler.step()
        np.random.shuffle(I)
        error = 0.0
        for i in range(0, sample_count, batch_size):
            print('Progress', round(100 * i / sample_count, 2), "%", end="\r")
            indices = I[i:i+batch_size]

            xBatch = utility.ToDevice(X[indices, :])
            yBatch = utility.ToDevice(Y[indices, :])

            prediction, matching_loss, vq_loss = network(
                xBatch, 
                k=torch.ones(1, device=xBatch.device), 
                t=yBatch
            )

            loss_function = nn.MSELoss()
            mse_loss = loss_function(utility.Normalize(yBatch, network.YNorm), utility.Normalize(prediction, network.YNorm))
            loss = mse_loss + matching_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            error += loss.item()

            loss_history.Add(
                (Item(mse_loss).item(), "MSE Loss"),
                (Item(matching_loss).item(), "Matching Loss"),
                # (Item(vq_loss).item(), "Quantization Loss"),
            )

            if loss_history.Counter == 0:
                
                network.eval()

                count = 100
                idx = random.choices(np.arange(sample_count), k=count)
                xBatch = utility.ToDevice(X[idx, :])
                yBatch = utility.ToDevice(Y[idx, :])
                reconstruction, _, _ = network(xBatch, k=torch.ones(1, device=xBatch.device), t=yBatch) #This is the result by reconstructing Y
                sampling, _ = network(xBatch, k=torch.ones(1, device=xBatch.device)) #This is the result by sampling Y

                ax2.clear()
                ax3.clear()
                ax4.clear()
                ax2.set_ylim(-4, 4)
                ax3.set_ylim(-4, 4)
                ax4.set_ylim(-4, 4)
                ax2.set_title("Ground Truth Y")
                ax3.set_title("Reconstruction (X,Y)->Y")
                ax4.set_title("Sampling X->Y")
                for i in range(len(idx)):
                    ax2.plot(yBatch[i].cpu())
                    ax3.plot(reconstruction[i].detach().cpu())
                    ax4.plot(sampling[i].detach().cpu())

                network.train()

                plt.gcf().canvas.draw()
                plt.gcf().canvas.start_event_loop(1e-2)

        print('Epoch', epoch+1, error/(sample_count/batch_size))
        loss_history.Print()

        utility.SaveONNX(
            path=save + '/' + str(epoch+1) + '.onnx',
            model=network,
            input_size=(torch.zeros(1, input_dim), torch.ones(1)),
            input_names=['X', 'K'],
            output_names=['Y', 'Code'],
            dynamic_axes={
                'K': {0: 'Size'}
            }
        )