import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#batches is a list of sequences of channels: list[sequence_length x channel_size]
#channels is the number of features per sample
def PCA2DSequence(ax, sequence_length, feature_dim, sequences, title):
    ax.cla()

    point_alpha = 0.3
    line_alpha = 0.2
    arrow_alpha = 1.0
    arrow_step = 10
    arrow_size = 0.015
    arrow_power = 1.0
    arrow_color = (0.25,0.25,0.5)

    indices = np.arange(len(sequences)*sequence_length)
    indices = indices.reshape(len(sequences), sequence_length)

    x = np.vstack(sequences).reshape(-1, feature_dim)
    y = PCA(n_components=2, whiten=True).fit_transform(x)
    for i in indices:
        _x_ = y[i]
        px = _x_[:,0]
        py = _x_[:,1]
        ax.plot(px, py, c=(0,0,0), alpha=line_alpha)
        ax.scatter(px, py, alpha=point_alpha)
        # for j in range(0, sequence_length-arrow_step, arrow_step):
        #     d = np.sqrt(px[j]**2 + py[j]**2)
        #     d = np.power(d, arrow_power)
        #     ax.arrow(px[j], py[j], px[j+1]-px[j], py[j+1]-py[j], alpha=arrow_alpha, width=d*arrow_size, color=arrow_color)

    ax.set_axis_off()
    ax.set_title(title)

def PlotFunctions(ax, values, title, yLimits=None):
    ax.cla()
    x = values.transpose(0,1)
    for i in range(len(x)):
        ax.plot(x[i])
    ax.set_title(title)
    if yLimits is not None:
        ax.set_ylim([yLimits[0], yLimits[1]])

def PlotGridMap(ax, values, title):
    ax.cla()
    ax.imshow(values, cmap=plt.cm.bone, aspect='auto')
    ax.set_axis_off()
    ax.set_title(title)
