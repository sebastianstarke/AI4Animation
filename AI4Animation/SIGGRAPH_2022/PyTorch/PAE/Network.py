import Library.Utility as utility
import Library.Plotting as plot
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler
import Network as this

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import random

import matplotlib.pyplot as plt

#Start Parameter Section
window = 2.0 #time duration of the time window
frames = 121 #sample count of the time window (60FPS)
keys = 13 #optional, used to rescale the FT window to resolution for motion controller training afterwards
joints = 24

input_channels = 3*joints #number of channels along time in the input data (here 3*J as XYZ-velocity component of each joint)
phase_channels = 8 #desired number of latent phase channels (usually between 2-10)

epochs = 10
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-4
restart_period = 10
restart_mult = 2

plotting_interval = 100 #update visualization at every n-th batch (visualization only)
pca_sequence_count = 10 #number of motion sequences visualized in the PCA (visualization only)
test_sequence_length = 150 #maximum length of each motion sequence (visualization only)
#End Parameter Section

if __name__ == '__main__':
        
    def Item(value):
        return value.detach().cpu()

    data_file = "../Dataset/Data.bin"
    shape = utility.LoadTxtAsInt("../Dataset/Shape.txt")
    sequences = utility.LoadTxtRaw("../Dataset/Sequences.txt")

    #Initialize visualization
    sequences = np.array(utility.Transpose2DList(sequences)[0], dtype=np.int64)
    sample_count = shape[0]
    feature_dim = shape[1]
    test_sequences = []
    for i in range(int(sequences[-1])):
        indices = np.where(sequences == (i+1))[0]
        intervals = int(np.floor(len(indices) / test_sequence_length))
        if intervals > 0:
            slices = np.array_split(indices, intervals)
            test_sequences += slices

    #Initialize all seeds
    seed = 23456
    rng = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    #Initialize drawing
    plt.ion()
    _, ax1 = plt.subplots(6,1)
    _, ax2 = plt.subplots(phase_channels,5)
    _, ax3 = plt.subplots(1,2)
    _, ax4 = plt.subplots(2,1)
    dist_amps = []
    dist_freqs = []
    loss_history = utility.PlottingWindow("Loss History", ax=ax4, min=0, drawInterval=plotting_interval)

    #Build network model
    network = utility.ToDevice(this.Model(
        input_channels=input_channels,
        embedding_channels=phase_channels,
        time_range=frames,
        key_range=keys,
        window=window
    ))

    #Setup optimizer and loss function
    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count, restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True)
    loss_function = torch.nn.MSELoss()

    I = np.arange(sample_count)
    for epoch in range(epochs):
        scheduler.step()
        rng.shuffle(I)
        for i in range(0, sample_count, batch_size):
            print('Progress', round(100 * i / sample_count, 2), "%", end="\r")
            train_indices = I[i:i+batch_size]

            #Run model prediction
            train_batch = utility.ToDevice(torch.from_numpy(utility.ReadBatch(data_file, train_indices, feature_dim)))
            yPred, latent, signal, params = network(train_batch)

            #Compute loss and train
            loss = loss_function(yPred, train_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            #Start Visualization Section
            _a_ = Item(params[2]).squeeze().numpy()
            for i in range(_a_.shape[0]):
                dist_amps.append(_a_[i,:])
            while len(dist_amps) > 10000:
                dist_amps.pop(0)

            _f_ = Item(params[1]).squeeze().numpy()
            for i in range(_f_.shape[0]):
                dist_freqs.append(_f_[i,:])
            while len(dist_freqs) > 10000:
                dist_freqs.pop(0)

            loss_history.Add(
                (Item(loss).item(), "Reconstruction Loss")
            )

            if loss_history.Counter == 0:
                plot.Functions(ax1[0], Item(train_batch[0]).reshape(network.input_channels,frames), -1.0, 1.0, -5.0, 5.0, title="Motion Curves" + " " + str(network.input_channels) + "x" + str(frames), showAxes=False)
                plot.Functions(ax1[1], Item(latent[0]), -1.0, 1.0, -2.0, 2.0, title="Latent Convolutional Embedding" + " " + str(phase_channels) + "x" + str(frames), showAxes=False)
                plot.Circles(ax1[2], Item(params[0][0]).squeeze(), Item(params[2][0]).squeeze(), title="Learned Phase Timing"  + " " + str(phase_channels) + "x" + str(2), showAxes=False)
                plot.Functions(ax1[3], Item(signal[0]), -1.0, 1.0, -2.0, 2.0, title="Latent Parametrized Signal" + " " + str(phase_channels) + "x" + str(frames), showAxes=False)
                plot.Functions(ax1[4], Item(yPred[0]).reshape(network.input_channels,frames), -1.0, 1.0, -5.0, 5.0, title="Curve Reconstruction" + " " + str(network.input_channels) + "x" + str(frames), showAxes=False)
                plot.Function(ax1[5], [Item(train_batch[0]), Item(yPred[0])], -1.0, 1.0, -5.0, 5.0, colors=[(0, 0, 0), (0, 1, 1)], title="Curve Reconstruction (Flattened)" + " " + str(1) + "x" + str(network.input_channels*frames), showAxes=False)
                plot.Distribution(ax3[0], dist_amps, title="Amplitude Distribution")
                plot.Distribution(ax3[1], dist_freqs, title="Frequency Distribution")

                test_batch = utility.ToDevice(torch.from_numpy(utility.ReadBatch(data_file, random.choice(test_sequences), feature_dim)))
                _, _, _, params = network(test_batch)

                for i in range(phase_channels):
                    phase = params[0][:,i]
                    freq = params[1][:,i]
                    amps = params[2][:,i]
                    offs = params[3][:,i]
                    plot.Phase1D(ax2[i,0], Item(phase), Item(amps), color=(0, 0, 0), title=("1D Phase Values" if i==0 else None), showAxes=False)
                    plot.Phase2D(ax2[i,1], Item(phase), Item(amps), title=("2D Phase Vectors" if i==0 else None), showAxes=False)
                    plot.Functions(ax2[i,2], Item(freq).transpose(0,1), -1.0, 1.0, 0.0, 4.0, title=("Frequencies" if i==0 else None), showAxes=False)
                    plot.Functions(ax2[i,3], Item(amps).transpose(0,1), -1.0, 1.0, 0.0, 1.0, title=("Amplitudes" if i==0 else None), showAxes=False)
                    plot.Functions(ax2[i,4], Item(offs).transpose(0,1), -1.0, 1.0, -1.0, 1.0, title=("Offsets" if i==0 else None), showAxes=False)
                
                #Manifold Computation and Visualization
                pca_indices = []
                pca_batches = []
                pivot = 0
                for i in range(pca_sequence_count):
                    indices = random.choice(test_sequences)
                    test_batch = utility.ToDevice(torch.from_numpy(utility.ReadBatch(data_file, indices, feature_dim)))
                    _, _, _, params = network(test_batch)
                    a = Item(params[2]).squeeze()
                    p = Item(params[0]).squeeze()
                    #Compute Phase Manifold (2D vectors composed of sin and cos terms)
                    m_x = a * np.sin(2.0 * np.pi * p)
                    m_y = a * np.cos(2.0 * np.pi * p)
                    manifold = torch.hstack((m_x, m_y))
                    pca_indices.append(pivot + np.arange(len(indices)))
                    pca_batches.append(manifold)
                    pivot += len(indices)

                plot.PCA2D(ax4[0], pca_indices, pca_batches, "Phase Manifold (" + str(pca_sequence_count) + " Random Sequences)")

                plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(1e-5)
            #End Visualization Section

        print('Epoch', epoch+1, loss_history.CumulativeValue())

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

    def atan2(self, y, x):
        tpi = self.tpi
        ans = torch.atan(y/x)
        ans = torch.where( (x<0) * (y>=0), ans+0.5*tpi, ans)
        ans = torch.where( (x<0) * (y<0), ans-0.5*tpi, ans)
        return ans

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
        y = torch.tanh(y)

        y = self.conv2(y)
        y = self.bn_conv2(y)
        y = torch.tanh(y)

        latent = y #Save latent for returning

        #Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        #Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            v = self.bn[i](v)
            p[:,i] = self.atan2(v[:,1], v[:,0]) / self.tpi

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
        y = self.bn_deconv1(y)
        y = torch.tanh(y)

        y = self.deconv2(y)

        y = y.reshape(y.shape[0], self.input_channels*self.time_range)

        return y, latent, signal, params