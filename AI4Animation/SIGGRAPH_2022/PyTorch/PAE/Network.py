import sys
sys.path.append("../")

import Library.Utility as utility
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler
import PAE as model
import Plotting as plot

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import random

import matplotlib.pyplot as plt

if __name__ == '__main__':

    #Start Parameter Section
    window = 2.0 #time duration of the time window
    fps = 60 #fps of the motion capture data
    joints = 26 #joints of the character skeleton

    frames = int(window * fps) + 1
    input_channels = 3*joints #number of channels along time in the input data (here 3*J as XYZ-component of each joint)
    phase_channels = 5 #desired number of latent phase channels (usually between 2-10)

    epochs = 10
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    plotting_interval = 500 #update visualization at every n-th batch (visualization only)
    pca_sequence_count = 100 #number of motion sequences visualized in the PCA (visualization only)
    test_sequence_ratio = 0.01 #ratio of randomly selected test sequences (visualization only)
    #End Parameter Section

    def LoadBatches(sequences):
        gather = gather_window.reshape(1,-1).repeat(sequences.shape[0], 0)

        sequences = data_sequences[sequences]

        pivot = sequences[:,0].reshape(-1,1)
        min = sequences[:,1].reshape(-1,1)
        max = sequences[:,2].reshape(-1,1)

        gather = np.clip(gather + pivot, min, max)

        shape = gather.shape

        batch = utility.ReadBatchFromMatrix(Data, gather.flatten())

        batch = batch.reshape(shape[0], shape[1], -1)
        batch = batch.swapaxes(1,2)
        batch = batch.reshape(shape[0], batch.shape[1]*batch.shape[2])
        return batch

    def Item(value):
        return value.detach().cpu()

    Load = "Dataset"
    Save = "Training"
    utility.MakeDirectory(Save)

    Data = Load+"/Data.bin"
    Shape = utility.LoadTxtAsInt(Load+"/DataShape.txt")
    Sequences = utility.LoadSequences(Load+"/Sequences.txt", True, Shape[0])

    # Sequences = utility.LoadSequences(Path+"/Sequences.txt", True, 100000)
    # Sequences = Sequences[np.where(Sequences == 1)]

    sample_count = Shape[0]
    feature_dim = Shape[1]
    gather_padding = (int((frames-1)/2))
    gather_window = np.arange(frames) - gather_padding

    #Pre-load Data Matrix
    Data = utility.ReadBinary(Data, sample_count, feature_dim)

    #Start Generate Data Sequences
    print("Generating Data Sequences")
    data_sequences = []
    test_sequences = []

    for i in range(Sequences[-1]):
        utility.PrintProgress(i, Sequences[-1])
        indices = np.where(Sequences == (i+1))[0]
        for j in range(indices.shape[0]):
            slice = [indices[j], indices[0], indices[-1]]
            data_sequences.append(slice)
            if np.random.uniform(0, 1) < test_sequence_ratio and indices[j] >= (indices[0]+gather_padding) and indices[j] <= (indices[-1]-gather_padding):
                test_sequences.append(j)

    print("Data Sequences:",len(data_sequences))
    print("Test Sequences:",len(test_sequences))
    data_sequences = np.array(data_sequences)
    sample_count = len(data_sequences)
    #End Generate Data Sequences

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
    network = utility.ToDevice(model.Model(
        input_channels=input_channels,
        embedding_channels=phase_channels,
        time_range=frames,
        window=window
    ))

    print("Training Phases")
    #Setup optimizer and loss function
    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count, restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True)
    loss_function = torch.nn.MSELoss()
    # loss_function = utility.GMSELoss()

    I = np.arange(sample_count)
    for epoch in range(epochs):
        scheduler.step()
        rng.shuffle(I)
        for i in range(0, sample_count, batch_size):
            utility.PrintProgress(i, sample_count, sample_count/batch_size)
            train_indices = I[i:i+batch_size]

            #Run model prediction
            network.train()
            train_batch = LoadBatches(train_indices)
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
                network.eval()

                plot.Functions(ax1[0], Item(train_batch[0]).reshape(network.input_channels,frames), -1.0, 1.0, -5.0, 5.0, title="Motion Curves" + " " + str(network.input_channels) + "x" + str(frames), showAxes=False)
                plot.Functions(ax1[1], Item(latent[0]), -1.0, 1.0, -2.0, 2.0, title="Latent Convolutional Embedding" + " " + str(phase_channels) + "x" + str(frames), showAxes=False)
                plot.Circles(ax1[2], Item(params[0][0]).squeeze(), Item(params[2][0]).squeeze(), title="Learned Phase Timing"  + " " + str(phase_channels) + "x" + str(2), showAxes=False)
                plot.Functions(ax1[3], Item(signal[0]), -1.0, 1.0, -2.0, 2.0, title="Latent Parametrized Signal" + " " + str(phase_channels) + "x" + str(frames), showAxes=False)
                plot.Functions(ax1[4], Item(yPred[0]).reshape(network.input_channels,frames), -1.0, 1.0, -5.0, 5.0, title="Curve Reconstruction" + " " + str(network.input_channels) + "x" + str(frames), showAxes=False)
                plot.Function(ax1[5], [Item(train_batch[0]), Item(yPred[0])], -1.0, 1.0, -5.0, 5.0, colors=[(0, 0, 0), (0, 1, 1)], title="Curve Reconstruction (Flattened)" + " " + str(1) + "x" + str(network.input_channels*frames), showAxes=False)
                plot.Distribution(ax3[0], dist_amps, title="Amplitude Distribution")
                plot.Distribution(ax3[1], dist_freqs, title="Frequency Distribution")

                indices = gather_window + random.choice(test_sequences)
                _, _, _, params = network(LoadBatches(indices))

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
                
                #Visualization
                pca_indices = []
                pca_batches = []
                pivot = 0
                for i in range(pca_sequence_count):
                    indices = gather_window + random.choice(test_sequences)
                    _, _, _, params = network(LoadBatches(indices))
                    a = Item(params[2]).squeeze()
                    p = Item(params[0]).squeeze()
                    b = Item(params[3]).squeeze()
                    m_x = a * np.sin(2.0 * np.pi * p) + b
                    m_y = a * np.cos(2.0 * np.pi * p) + b
                    manifold = torch.hstack((m_x, m_y))
                    pca_indices.append(pivot + np.arange(len(indices)))
                    pca_batches.append(manifold)
                    pivot += len(indices)

                plot.PCA2D(ax4[0], pca_indices, pca_batches, "Phase Manifold (" + str(pca_sequence_count) + " Random Sequences)")

                plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(1e-5)
            #End Visualization Section

        torch.save(network, Save + "/"+str(epoch+1)+"_"+str(phase_channels)+"Channels"+".pt") 

        print('Epoch', epoch+1, loss_history.CumulativeValue())

        #Save Phase Parameters
        print("Saving Parameters")
        network.eval()
        E = np.arange(sample_count)
        with open(Save+'/Parameters_'+str(epoch+1)+'.txt', 'w') as file:
            for i in range(0, sample_count, batch_size):
                utility.PrintProgress(i, sample_count)
                eval_indices = E[i:i+batch_size]
                eval_batch = LoadBatches(eval_indices)
                _, _, _, params = network(eval_batch)
                p = utility.ToNumpy(params[0]).squeeze()
                f = utility.ToNumpy(params[1]).squeeze()
                a = utility.ToNumpy(params[2]).squeeze()
                b = utility.ToNumpy(params[3]).squeeze()
                for j in range(p.shape[0]):
                    params = np.concatenate((p[j,:],f[j,:],a[j,:],b[j,]))
                    line = ' '.join(map(str, params))
                    if (i+j) == (sample_count-1):
                        file.write(line)
                    else:
                        file.write(line + '\n')
