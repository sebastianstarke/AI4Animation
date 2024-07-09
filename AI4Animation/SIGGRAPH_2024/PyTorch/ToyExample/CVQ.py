import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import einsum
from einops import rearrange
# import torchsort

class VectorQuantizer(nn.Module):
    """
    Improved version over vector quantizer, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, embed_channels, embed_dim, beta, distance='cos', anchor='probrandom', first_batch=False, contras_loss=False):
        super().__init__()

        self.num_embed = embed_channels*embed_dim
        self.embed_channels = embed_channels
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.K = embed_channels*embed_dim
        self.C = embed_channels
        self.D = embed_dim

        try:
            self.size = int(math.pow(self.K, self.C))
        except OverflowError:
            self.size = "Larger than int64"

        print("CVQ Quantization")
        print("Codebook Size:",self.K)
        print("Codebook Channels:",self.C)
        print("Codebook Dimensions:",self.D)
        print("Codebook Capacity:",self.size)

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))

    def forward(self, z):
        z_flattened = z.view(-1, self.embed_dim)

        # calculate the distance
        if self.distance == 'l2':
            # Compute L2 distance between latents and embedding weights
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.matmul(z_flattened.detach(), self.embedding.weight.t())

        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))

        if self.training:
            # encoding by closest point in the embedding
            encoding_indices = torch.argmax(d, dim=1)

            # quantise and unflatten
            z_q = self.embedding.weight[encoding_indices].view(z.shape)

            # compute loss for embedding
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)

            # preserve gradients
            z_q = z + (z_q - z).detach()

            # online clustered reinitialisation for unoptimized points
            encodings = torch.zeros(encoding_indices.shape[0], self.num_embed, device=z.device)
            encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
            avg_probs = torch.mean(encodings, dim=0)
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1,:]]
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss += contra_loss
        else:
            # distances, indices = self.topk(d, knn.size(0))
            # encoding_indices = torch.cat(indices).reshape(knn.size(0), -1)
            # z_q = self.embedding.weight[encoding_indices].reshape(knn.size(0), -1)

            encoding_indices = torch.argmax(d, dim=1)
            z_q = self.embedding.weight[encoding_indices].view(z.shape)
            loss = None
                
        return z_q, loss

    def topk(self, d, k):
        index = []
        distance = []
        index.append(torch.argmax(d, dim=1))
        distance.append(torch.gather(d, dim=1, index=index[-1].reshape(-1,1)).flatten())
        for i in range(k-1):
            indices = torch.cat((torch.arange(index[-1].shape[0], device=d.device), index[-1]), dim=0).reshape(2,-1).transpose(0,1)
            d.index_put_(indices=tuple(indices.t()), values=torch.tensor(torch.finfo(torch.float32).min))
            index.append(torch.argmax(d, dim=1))
            distance.append(torch.gather(d, dim=1, index=index[-1].reshape(-1,1)).flatten())
        return distance, index

class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features
    



    # def topk(self, d, k):
    #     index = []
    #     distance = []
    #     index.append(torch.argmax(d, dim=1))
    #     distance.append(torch.gather(d, dim=1, index=index[-1].reshape(-1,1)).flatten())
    #     for i in range(k-1):
    #         indices = torch.cat((torch.arange(index[-1].shape[0], device=d.device), index[-1]), dim=0).reshape(2,-1).transpose(0,1)
    #         d.index_put_(indices=tuple(indices.t()), values=torch.tensor(torch.finfo(torch.float32).min))
    #         index.append(torch.argmax(d, dim=1))
    #         distance.append(torch.gather(d, dim=1, index=index[-1].reshape(-1,1)).flatten())
    #     return distance, index



            # # encoding by categorical distribution
            # levels = 2
            # distances, indices = self.topk(d, levels)
            # probs = torch.cat(distances, dim=0).reshape(levels,-1)
            # probs = probs.reshape(-1, levels)
            # probs = F.softmax(probs, dim=1)
            # dist = torch.distributions.categorical.Categorical(probs=probs)
            # idx = dist.sample()
            # inds = torch.cat(indices, dim=0).reshape(levels,-1)
            # inds = inds.reshape(-1, levels)
            # sample = torch.gather(inds, dim=1, index=idx.reshape(-1,1))
            # sample = sample.flatten()
            # z_q = self.embedding.weight[sample].view(z.shape)
            # return z_q



            # #encoding by bernoulli sampling
            # #generates a set of kNN codes based on sampling probabilities p=[0,1]
            # #the probability controls how many indices from the closest code are switched with more distant codes
            # #p is a tensor of probabilities where p=0.0 is the closest code and p=1.0 is the furthest code that can be sampled
            # #default usage can be achieved by passing torch.tensor([0.0])
            # #kNN samples can be generated by passing torch.tensor([0.0, 0.7, 0.2, ..., 0.4])
            # distances, indices = self.topk(d,2)
            # x = indices[0]
            # y = indices[1]
            # probs = prob.repeat(self.C, 1).reshape(-1,prob.size(0)).transpose(0,1)
            # mask = torch.bernoulli(probs).to(torch.int64)
            # sample = (1-mask)*x + mask*y
            # z_q = self.embedding.weight[sample].reshape(prob.size(0),-1)
            # return z_q