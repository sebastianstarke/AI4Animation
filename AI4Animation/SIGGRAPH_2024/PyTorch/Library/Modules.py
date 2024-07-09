import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 K: int,
                 C: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = K
        self.C = C
        self.D = int(K/C)
        self.beta = beta

        try:
            self.size = int(math.pow(self.K, self.C))
        except OverflowError:
            self.size = "Larger than int64"

        print("VQ Quantization")
        print("Codebook Codebook Size:",self.K)
        print("Codebook Codebook Channels:",self.C)
        print("Codebook Capacity:",self.size)

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents, k): #k is the indices of the kNN for each channel. Default: [0,0,...,0]
        latents = latents.contiguous()
        flat_latents = latents.view(-1, self.D)  # [B x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [B x K]

        # Get the encoding that has the min distance
        # encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)

        # Get the encoding with the k-th min distance
        # if self.training:
        #     encoding_inds = torch.argsort(dist, dim=1, descending=False)[:,random.randint(0,5)].unsqueeze(1)
        # else:
        #     encoding_inds = torch.argsort(dist, dim=1, descending=False)[:,0].unsqueeze(1)

        # Get the encoding with the k-th min distance
        encoding_inds = torch.argsort(dist, dim=1, descending=False)
        encoding_inds = encoding_inds.reshape(-1, self.C, self.K)
        k = k.to(torch.int64)
        select = torch.zeros(encoding_inds.shape[0], encoding_inds.shape[1], dtype=torch.int64, device=encoding_inds.device)
        for i in range(encoding_inds.shape[1]):
            select[:,i] = encoding_inds[:,i,k[i]]
        encoding_inds = select.reshape(-1, 1)

        # Convert to one-hot encodings
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=latents.device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [-1 x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [B, D]
        quantized_latents = quantized_latents.view(latents.shape)  # [B x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.contiguous(), vq_loss
    
class ArgMaxQuantizer(nn.Module):
    def __init__(self,
                 K: int,
                 C: int):
        super(ArgMaxQuantizer, self).__init__()
        self.K = K
        self.C = C

        self.size = int(math.pow(self.K, self.C))

        print("Codebook Codebook Size:",self.K)
        print("Codebook Codebook Channels:",self.C)
        print("Codebook Capacity:",self.size)

    def forward(self, latents, k): #k is the indices of the kNN for each channel. Default: [0,0,...,0]
        latents = latents.contiguous()

        # Get the encoding with the k-th highest activation
        values = latents.reshape(-1, self.C, self.K)

        values = F.softmax(values, dim=2)
        
        encoding_inds = torch.argsort(values, dim=2, descending=True)
        encoding_inds = encoding_inds.reshape(-1, self.C, self.K)
        k = k.to(torch.int64)
        select = torch.zeros(encoding_inds.shape[0], encoding_inds.shape[1], dtype=torch.int64, device=encoding_inds.device)
        for i in range(encoding_inds.shape[1]):
            select[:,i] = encoding_inds[:,i,k[i]]
        encoding_inds = select.reshape(-1, 1)

        # Convert to one-hot encodings
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=latents.device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [-1 x K]
        encoding_one_hot = encoding_one_hot.reshape(-1, self.C*self.K)

        # Add the residue back to the latents
        values = values.reshape(-1, self.C*self.K)
        encoding_one_hot = values + (encoding_one_hot - values).detach()

        return encoding_one_hot.contiguous()

class ArgMaxSelector(nn.Module):
    def __init__(self, K: int):
        super(ArgMaxSelector, self).__init__()
        self.K = K

    def forward(self, latents, k):
        k = k[0].to(torch.int64) #Index of kNN. Default: 0

        latents = latents.contiguous()

        encoding_inds = torch.argmax(latents, dim=1).unsqueeze(1)

        # encoding_inds = torch.argsort(latents, dim=1, descending=True)[:,k].unsqueeze(1)
        
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=latents.device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)
        encoding_one_hot = latents + (encoding_one_hot - latents).detach()

        return encoding_one_hot.contiguous()

# class GaussianSampler(torch.nn.Module):
#     def __init__(self, input_size, samples_size):
#         super(GaussianSampler, self).__init__()

#         self.Mu = nn.Linear(input_size, samples_size)
#         self.LogVar = nn.Linear(input_size, samples_size)
        
#     def forward(self, input):
#         mu = self.Mu(input)
#         logvar = self.LogVar(input)

#         if self.training:
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             z = eps * std + mu
#         else:
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             z = eps * std + mu

#         kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim = 1), dim = 0)
#         return z, kld_loss

class VariationalEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, samples_size, dropout):
        super(VariationalEncoder, self).__init__()

        self.dropout = dropout

        self.L1 = nn.Linear(input_size, hidden1_size)
        self.L2 = nn.Linear(hidden1_size, hidden2_size)
        self.Mu = nn.Linear(hidden2_size, samples_size)
        self.LogVar = nn.Linear(hidden2_size, samples_size)
        
    def forward(self, input):
        z = input

        z = F.dropout(z, self.dropout, training=self.training)
        z = self.L1(z)
        z = F.elu(z)

        z = F.dropout(z, self.dropout, training=self.training)
        z = self.L2(z)
        z = F.elu(z)

        z = F.dropout(z, self.dropout, training=self.training)
        mu = self.Mu(z)
        logvar = self.LogVar(z)

        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        # z = eps * std + mu

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim = 1), dim = 0)

        # return z, kld_loss

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu

            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim = 1), dim = 0)

            return z, kld_loss
        else:
            return mu, torch.tensor(0.0)

class LinearEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, dropout):
        super(LinearEncoder, self).__init__()

        self.InputSize = input_size
        self.OutputSize = output_size

        self.Dropout = dropout

        self.L1 = nn.Linear(input_size, hidden1_size)
        self.L2 = nn.Linear(hidden1_size, hidden2_size)
        self.L3 = nn.Linear(hidden2_size, output_size)
        
    def forward(self, z):
        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L1(z)
        z = F.elu(z)

        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L2(z)
        z = F.elu(z)

        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L3(z)

        return z

class LinearFiLMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, film_size, dropout):
        super(LinearFiLMEncoder, self).__init__()

        self.InputSize = input_size
        self.Hidden1Size = hidden1_size
        self.Hidden2Size = hidden2_size
        self.OutputSize = output_size
        self.FiLMSize = film_size

        self.Dropout = dropout

        self.L1 = nn.Linear(input_size, hidden1_size)
        self.L2 = nn.Linear(hidden1_size, hidden2_size)
        self.L3 = nn.Linear(hidden2_size, output_size)

        self.F1Scale = LinearEncoder(film_size, input_size, input_size, input_size, dropout)
        self.F1Shift = LinearEncoder(film_size, input_size, input_size, input_size, dropout)

        self.F2Scale = LinearEncoder(film_size, hidden1_size, hidden1_size, hidden1_size, dropout)
        self.F2Shift = LinearEncoder(film_size, hidden1_size, hidden1_size, hidden1_size, dropout)

        self.F3Scale = LinearEncoder(film_size, hidden2_size, hidden2_size, hidden2_size, dropout)
        self.F3Shift = LinearEncoder(film_size, hidden2_size, hidden2_size, hidden2_size, dropout)
        
    def forward(self, z, film):

        z = self.F1Scale(film) * z + self.F1Shift(film)
        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L1(z)
        z = F.elu(z)

        z = self.F2Scale(film) * z + self.F2Shift(film)
        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L2(z)
        z = F.elu(z)

        z = self.F3Scale(film) * z + self.F3Shift(film)
        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L3(z)

        return z

class GatedExpertsEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gating_indices, gating_hidden, experts, dropout):
        super(GatedExpertsEncoder, self).__init__()

        self.InputSize = input_size
        self.HiddenSize = hidden_size
        self.OutputSize = output_size
        self.Experts = experts
        self.Dropout = dropout
        self.Indices = gating_indices

        self.G = LinearEncoder(len(gating_indices), gating_hidden, gating_hidden, experts, dropout)
        self.L = LinearExpertsEncoder(input_size, hidden_size, hidden_size, output_size, experts, dropout)

    def forward(self, z):
        w = self.G(z[:, self.Indices])
        z = self.L(z, w)
        return z

class LinearExpertsEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, experts, dropout):
        super(LinearExpertsEncoder, self).__init__()

        self.InputSize = input_size
        self.Hidden1Size = hidden1_size
        self.Hidden2Size = hidden2_size
        self.OutputSize = output_size
        self.Experts = experts
        self.Dropout = dropout

        self.L1 = LinearExpertLayer(experts, input_size, hidden1_size)
        self.L2 = LinearExpertLayer(experts, hidden1_size, hidden2_size)
        self.L3 = LinearExpertLayer(experts, hidden2_size, output_size)

    def forward(self, z, weights):
        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L1(z, weights)
        z = F.elu(z)

        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L2(z, weights)
        z = F.elu(z)

        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L3(z, weights)

        return z

#Output-Blended MoE Layer
class LinearExpertLayer(torch.nn.Module):
    def __init__(self, experts, input_dim, output_dim):
        super(LinearExpertLayer, self).__init__()

        self.experts = experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = self.weights([experts, input_dim, output_dim])
        self.b = self.bias([experts, 1, output_dim])

    def forward(self, x, w):
        y = torch.zeros((x.shape[0], self.output_dim), device=x.device, requires_grad=True)
        for i in range(self.experts):
            y = y + w[:,i].unsqueeze(1) * (x.matmul(self.W[i,:,:]) + self.b[i,:,:])
        return y

    def weights(self, shape):
        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha = np.asarray(np.random.uniform(low=-alpha_bound, high=alpha_bound, size=shape), dtype=np.float32)
        return Parameter(torch.from_numpy(alpha), requires_grad=True)

    def bias(self, shape):
        return Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=True)

class ConvolutionalEncoder(torch.nn.Module):
    def __init__(self, input_dim, window_size, input_channels, hidden_channels, output_channels, output_dim, dropout):
        super(ConvolutionalEncoder, self).__init__()

        self.dropout = dropout

        self.window_size = window_size
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.E = nn.Linear(input_dim, input_channels * window_size)
        self.C1 = nn.Conv1d(input_channels, hidden_channels, window_size, stride=1, padding=int((window_size - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.C2 = nn.Conv1d(hidden_channels, output_channels, window_size, stride=1, padding=int((window_size - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.D = nn.Linear(output_channels * window_size, output_dim)
        
    def forward(self, input):
        z = input

        z = self.E(z)
        z = z.reshape(-1, self.input_channels, self.window_size)

        z = F.dropout(z, self.dropout, training=self.training)
        z = self.C1(z)
        z = F.elu(z)

        z = F.dropout(z, self.dropout, training=self.training)
        z = self.C2(z)
        z = F.elu(z)

        z = z.reshape(-1, self.output_channels * self.window_size)
        z = self.D(z)

        return z