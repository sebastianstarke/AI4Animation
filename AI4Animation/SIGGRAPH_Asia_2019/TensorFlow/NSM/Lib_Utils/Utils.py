import os.path
import numpy as np


def build_path(path):
    for i in path:
        if not os.path.exists(i):
            os.makedirs(i)


def Normalize_byAxis(X, axis, savefile=None):
    Xmean, Xstd = X.mean(axis=axis), X.std(axis=axis)
    for i in range(Xstd.size):
        if (Xstd[i] == 0):
            Xstd[i] = 1
    X = (X - Xmean) / Xstd
    if savefile != None:
        Xmean.tofile(savefile + 'mean.bin')
        Xstd.tofile(savefile + 'std.bin')
    return X, Xmean, Xstd


def Normalize(X, N, savefile=None):
    mean = N[0]
    std = N[1]
    for i in range(std.size):
        if std[i] == 0:
            std[i] = 1
    if savefile != None:
        mean.tofile(savefile + 'mean.bin')
        std.tofile(savefile + 'std.bin')
    return (X - mean) / std, mean, std


def Normalize_Except(X, N, list_unNorm, savefile=None):
    mean = N[0]
    std = N[1]
    for i in range(len(list_unNorm)):
        mean[list_unNorm[i]] = 0
        std[list_unNorm[i]] = 1
    for i in range(std.size):
        if std[i] == 0:
            std[i] = 1
    if savefile != None:
        mean.tofile(savefile + 'mean.bin')
        std.tofile(savefile + 'std.bin')
    return (X - mean) / std, mean, std


def Gaussian_Noise(x, std, dims=None):
    if std == 0:
        return x
    elif dims == None:
        return x + np.random.normal(0.0, std, (x.shape[0], x.shape[1])).astype(np.float32)
    else:
        noise = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
        noise[:, dims] = np.random.normal(0.0, std, (x.shape[0], len(dims))).astype(np.float32)
        return x + noise


def getNormalizedOffset(input_data, output_data, index_input, index_output, savefile):
    offset = output_data[:, index_output] - input_data[:, index_input]
    output_data[:, index_output] = offset
    mean = output_data.mean(axis=0)
    std = output_data.std(axis=0)
    for i in range(std.size):
        if std[i] == 0:
            std[i] = 1
    mean.tofile(savefile + 'mean.bin')
    std.tofile(savefile + 'std.bin')
    return (output_data - mean) / std


def get2dIndex(list1d, size_window):
    """
    Given a 1d list which is the beginning index of one window
    :param list1d: list of beginning indices of windows
    :param size_window:
    :return: 2d int array of indices
    """
    step = int(size_window / 2)
    num_window = len(list1d)
    results = np.zeros((num_window, size_window), dtype=np.int)
    for i in range(num_window):
        results[i] = np.arange(list1d[i] - step, list1d[i] + step + 1)
    return results


def get_index_train(sequence_index, len_sequence):
    """
    give the sequence.txt which tells the corresponding sequence for each frame, and the length of target window
    out the valid index for starting sample point
    """
    index_train = []
    for i in range(len(sequence_index) - len_sequence):
        if sequence_index[i] == sequence_index[i + len_sequence]:
            index_train += [i]
    return np.array(index_train, dtype=np.int32)


def get_index_batch(index, len_sequence):
    """
    give the index of starting and length of sampling, return 2D indexes
    """
    index_final = []
    for i in range(index.shape[0]):
        index_final.append(np.arange(index[i], index[i] + len_sequence))
    index_final = np.asarray(index_final)
    return index_final


# HARD CODING FOR NORMALIZE THE HEIGHT MAP DIFFERENTLY
def NormalizeMINMAX(X, N, DimMINMAX, savefile=None):
    mean = N[0]
    std = N[1]

    if len(DimMINMAX) == 0:
        DimMINMAX = np.arange(len(mean))

    for i in range(std.size):
        if std[i] == 0:
            std[i] = 1

    X_MINMAX = X[:, DimMINMAX]
    min_norm = X_MINMAX.min()
    max_norm = X_MINMAX.max()
    mean_minmax = min_norm
    std_minmax = max_norm - min_norm
    mean[DimMINMAX] = mean_minmax
    std[DimMINMAX] = std_minmax
    if savefile != None:
        mean.tofile(savefile + 'mean.bin')
        std.tofile(savefile + 'std.bin')
    return (X - mean) / std


def NormalizeNP(X, N, DimMINMAX, savefile=None):
    mean = N[0]
    std = N[1]

    if len(DimMINMAX) == 0:
        DimMINMAX = np.arange(len(mean))

    for i in range(std.size):
        if std[i] == 0:
            std[i] = 1
    X_MINMAX = X[:, DimMINMAX]
    min_norm = X_MINMAX.min()
    max_norm = X_MINMAX.max()
    mean_minmax = (max_norm - min_norm) / 2 + min_norm
    std_minmax = max_norm - mean_minmax
    mean[DimMINMAX] = mean_minmax
    std[DimMINMAX] = std_minmax
    if savefile != None:
        mean.tofile(savefile + 'mean.bin')
        std.tofile(savefile + 'std.bin')
    return (X - mean) / std


def NormalizeMean(X, N, DimMINMAX, savefile=None):
    mean = N[0]
    std = N[1]

    if len(DimMINMAX) == 0:
        DimMINMAX = np.arange(len(mean))

    for i in range(std.size):
        if std[i] == 0:
            std[i] = 1
    X_MINMAX = X[:, DimMINMAX]
    mean_norm = X_MINMAX.mean()
    max_norm = X_MINMAX.max()
    mean_minmax = mean_norm
    std_minmax = max_norm - mean_norm
    mean[DimMINMAX] = mean_minmax
    std[DimMINMAX] = std_minmax
    if savefile != None:
        mean.tofile(savefile + 'mean.bin')
        std.tofile(savefile + 'std.bin')
    return (X - mean) / std


def Normalize_Global(X, N, DimMINMAX, savefile=None):
    mean = N[0]
    std = N[1]

    if len(DimMINMAX) == 0:
        DimMINMAX = np.arange(len(mean))

    for i in range(std.size):
        if std[i] == 0:
            std[i] = 1
    X_MINMAX = X[:, DimMINMAX]
    mean_norm = X_MINMAX.mean()
    std_norm = X_MINMAX.std()
    mean[DimMINMAX] = mean_norm
    std[DimMINMAX] = std_norm

    if savefile != None:
        mean.tofile(savefile + 'mean.bin')
        std.tofile(savefile + 'std.bin')
    return (X - mean) / std
