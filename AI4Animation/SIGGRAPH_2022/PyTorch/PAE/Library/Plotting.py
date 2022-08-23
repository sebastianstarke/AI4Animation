import numpy as np
import torch
from sklearn.decomposition import PCA

def PCA2D(ax, indices, batches, title):
    ax.cla()

    point_alpha = 0.3
    line_alpha = 0.2
    arrow_alpha = 1.0
    arrow_step = 50
    arrow_size = 0.015
    arrow_power = 1.0
    arrow_color = (0.25,0.25,0.5)

    x = np.vstack(batches)
    y = PCA(n_components=2, whiten=True).fit_transform(x)
    for i in indices:
        _x_ = y[i]
        px = _x_[:,0]
        py = _x_[:,1]
        if line_alpha != 0.0:
            ax.plot(px, py, c=(0,0,0), alpha=line_alpha)
        if point_alpha != 0.0:
            ax.scatter(px, py, alpha=point_alpha)
        if arrow_alpha != 0.0:
            for j in range(0, len(_x_)-arrow_step, arrow_step):
                d = np.sqrt(px[j]**2 + py[j]**2)
                d = np.power(d, arrow_power)
                ax.arrow(px[j], py[j], px[j+1]-px[j], py[j+1]-py[j], alpha=arrow_alpha, width=d*arrow_size, color=arrow_color)

    ax.set_axis_off()
    ax.set_title(title)

def Distribution(ax, values, title):
    ax.cla()
    
    means = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    args = np.arange(len(means))
    labels = ["Channel " + str(i+1) for i in args]
    ax.bar(np.arange(len(means)), means, yerr=std,
        align='center',
        alpha=0.5,
        ecolor='black',
        capsize=10)
    ax.set_xticks(args)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.yaxis.grid(True)

def Circles(ax, P, M, title=None, showAxes=True):
    ax.cla()

    aspect = 0.5
    ax.set_aspect(aspect)
    count = P.shape[0]
    ax.set_xlim(0, count+1)        
    ax.set_ylim(-1, 1)
    theta = np.linspace(0, 2*np.pi, 100)

    for i in range(count):
        p = P[i]
        m = M[i]
        x1 = aspect*m*np.cos(theta) + i + 1
        x2 = m*np.sin(theta)
        ax.plot(x1, x2)
        lineX = [i+1, i+1 + aspect*m*np.cos(2*np.pi*p)]
        lineY = [0, m*np.sin(2*np.pi*p)]
        ax.plot(lineX, lineY, color=(0,0,0))

    if title != None:
        ax.set_title(title)
    if showAxes == False:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

def Function(ax, list, xmin, xmax, ymin, ymax, colors=None, title=None, showAxes=True):
    ax.cla()
    idx = 0
    for values in list:
        args = np.linspace(xmin, xmax, values.shape[0])
        if colors != None:
            ax.plot(args, values, color=colors[idx])
        else:
            ax.plot(args, values)
        idx += 1
    ax.set_ylim(ymin, ymax)
    if title != None:
        ax.set_title(title)
    if showAxes == False:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

def Functions(ax, values, xmin, xmax, ymin, ymax, colors=None, title=None, showAxes=True):
    ax.cla()
    idx = 0
    args = np.linspace(xmin, xmax, values.shape[1])
    for i in range(values.shape[0]):
        if colors != None:
            ax.plot(args, values[i,:], color=colors[idx])
        else:
            ax.plot(args, values[i,:])
    idx += 1
    ax.set_ylim(ymin, ymax)
    if title != None:
        ax.set_title(title)
    if showAxes == False:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

def Phase1D(ax, phase, opacity, color=None, title=None, showAxes=True):
    ax.cla()
    phase = np.where(phase < 0, phase, phase+1)
    phase = phase % 1.0
    args = np.arange(len(phase))
    opacity = opacity.numpy().flatten()
    opacity = np.clip(opacity, 0, 1) #Avoid values <0 or >1
    for i in range(1, len(phase)):
        ax.plot([args[i-1], args[i]], [phase[i-1], phase[i]], color=(color[0], color[1], color[2], opacity[i]))
    ax.set_ylim(0,1)
    if title != None:
        ax.set_title(title)
    if showAxes == False:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

def Phase2D(ax, P, M, colors=None, title=None, showAxes=True):
    ax.cla()
    args = np.arange(len(P))

    phaseX = M*torch.sin(2*np.pi*P)
    phaseY = M*torch.cos(2*np.pi*P)

    if colors == None:
        ax.plot(args, phaseX)
        ax.plot(args, phaseY)
    else:
        ax.plot(args, phaseX, color=colors[0])
        ax.plot(args, phaseY, color=colors[1])
    ax.set_ylim(-1.0,1.0)
    if title != None:
        ax.set_title(title)
    if showAxes == False:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)