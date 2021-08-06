import os.path
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def BuildPath(path):
    for i in path:
        if not os.path.exists(i):
            os.makedirs(i)

def CollectPointers(file, max=0):
    pointers = []
    with open(file) as f:
        pivot = 0
        while(f.readline()):
            pointers.append(pivot)
            pivot = f.tell()
            if len(pointers) % 10000 == 0:
                print("Lines Read:", len(pointers))
            if max > 0 and len(pointers) == max:
                break
    return pointers

def ReadChunk(file, pointers):
    data = []
    with open(file) as f:
        for i in pointers:
            f.seek(i)
            data.append(np.float32(np.fromstring(f.readline(), sep=' ')))
    return np.concatenate(data).reshape(len(pointers), -1)

def ReadData(file):
    return np.float32(np.loadtxt(file))

def ReadNorm(file):
    N = np.float32(np.loadtxt(file))
    N[1, N[1, :] == 0] = 1
    return N

def Normalize(X, N):
    mean = N[0]
    std = N[1]
    return (X - mean) / std

def GaussianNoise(x, std, dims=None):
    if std==0:
        return x
    elif dims==None:
        return x + np.random.normal(0.0, std, (x.shape[0], x.shape[1])).astype(np.float32)
    else:
        noise = np.zeros((x.shape[0], x.shape[1]),dtype=np.float32)
        noise[:,dims] = np.random.normal(0.0, std, (x.shape[0], len(dims))).astype(np.float32)
        return x + noise

def ScaleLoss(sub, scale):
    if scale:
        scale = np.array(scale)
        if(len(scale.shape)<2):
            scale = np.expand_dims(scale, 0)
            scale = tf.constant(scale, dtype=tf.float32)
            return sub*scale
    else:
        return sub

def ComputeLoss(yTrue, yPred, dims, types, sess):
    loss = tf.Variable(0.0)
    for i, id in enumerate(types):
        if id == "MSE":
            loss += tf.reduce_sum(tf.square(tf.gather(yTrue, dims[i], axis=1) - tf.gather(yPred, dims[i], axis=1)))
        if id == "LOG":
            loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(tf.gather(yTrue, dims[i], axis=1), tf.gather(yPred, dims[i], axis=1)))
        if id == "COS":
            loss += tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(yTrue,0), tf.nn.l2_normalize(yPred,0)))
    loss /= len(dims)
    return loss
