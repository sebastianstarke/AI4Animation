import os.path

def build_path(path):
    for i in path:
        if not os.path.exists(i):
            os.makedirs(i)
    

def Normalize(X, axis, savefile= None):
    Xmean, Xstd = X.mean(axis=axis), X.std(axis=axis)
    for i in range(Xstd.size):
        if (Xstd[i]==0):
            Xstd[i]=1
    X = (X - Xmean) / Xstd
    if savefile != None:
        Xmean.tofile(savefile+'mean.bin')
        Xstd.tofile(savefile+'std.bin')
    return X
