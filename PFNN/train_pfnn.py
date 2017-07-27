import sys
import numpy as np
import theano
import theano.tensor as T
theano.config.allow_gc = True

sys.path.append('./nn')

from Layer import Layer
from HiddenLayer import HiddenLayer
from BiasLayer import BiasLayer
from DropoutLayer import DropoutLayer
from ActivationLayer import ActivationLayer
from AdamTrainer import AdamTrainer

rng = np.random.RandomState(23456)

""" Load Data """

database = np.load('database.npz')
X = database['Xun'].astype(theano.config.floatX)
Y = database['Yun'].astype(theano.config.floatX)
P = database['Pun'].astype(theano.config.floatX)

print(X.shape, Y.shape)

""" Calculate Mean and Std """

Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)

j = 31
w = ((60*2)//10)

Xstd[w*0:w* 1] = Xstd[w*0:w* 1].mean() # Trajectory Past Positions
Xstd[w*1:w* 2] = Xstd[w*1:w* 2].mean() # Trajectory Future Positions
Xstd[w*2:w* 3] = Xstd[w*2:w* 3].mean() # Trajectory Past Directions
Xstd[w*3:w* 4] = Xstd[w*3:w* 4].mean() # Trajectory Future Directions
Xstd[w*4:w*10] = Xstd[w*4:w*10].mean() # Trajectory Gait

""" Mask Out Unused Joints in Input """

joint_weights = np.array([
    1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10]).repeat(3)

Xstd[w*10+j*3*0:w*10+j*3*1] = Xstd[w*10+j*3*0:w*10+j*3*1].mean() / (joint_weights * 0.1) # Pos
Xstd[w*10+j*3*1:w*10+j*3*2] = Xstd[w*10+j*3*1:w*10+j*3*2].mean() / (joint_weights * 0.1) # Vel
Xstd[w*10+j*3*2:          ] = Xstd[w*10+j*3*2:          ].mean() # Terrain

Ystd[0:2] = Ystd[0:2].mean() # Translational Velocity
Ystd[2:3] = Ystd[2:3].mean() # Rotational Velocity
Ystd[3:4] = Ystd[3:4].mean() # Change in Phase
Ystd[4:8] = Ystd[4:8].mean() # Contacts

Ystd[8+w*0:8+w*1] = Ystd[8+w*0:8+w*1].mean() # Trajectory Future Positions
Ystd[8+w*1:8+w*2] = Ystd[8+w*1:8+w*2].mean() # Trajectory Future Directions

Ystd[8+w*2+j*3*0:8+w*2+j*3*1] = Ystd[8+w*2+j*3*0:8+w*2+j*3*1].mean() # Pos
Ystd[8+w*2+j*3*1:8+w*2+j*3*2] = Ystd[8+w*2+j*3*1:8+w*2+j*3*2].mean() # Vel
Ystd[8+w*2+j*3*2:8+w*2+j*3*3] = Ystd[8+w*2+j*3*2:8+w*2+j*3*3].mean() # Rot

""" Save Mean / Std / Min / Max """

Xmean.astype(np.float32).tofile('./demo/network/pfnn/Xmean.bin')
Ymean.astype(np.float32).tofile('./demo/network/pfnn/Ymean.bin')
Xstd.astype(np.float32).tofile('./demo/network/pfnn/Xstd.bin')
Ystd.astype(np.float32).tofile('./demo/network/pfnn/Ystd.bin')

""" Normalize Data """

X = (X - Xmean) / Xstd
Y = (Y - Ymean) / Ystd

""" Phase Function Neural Network """

class PhaseFunctionedNetwork(Layer):
    
    def __init__(self, rng=rng, input_shape=1, output_shape=1, dropout=0.7):
        
        self.nslices = 4        
        self.dropout0 = DropoutLayer(dropout, rng=rng)
        self.dropout1 = DropoutLayer(dropout, rng=rng)
        self.dropout2 = DropoutLayer(dropout, rng=rng)
        self.activation = ActivationLayer('ELU')
        
        self.W0 = HiddenLayer((self.nslices, 512, input_shape-1), rng=rng, gamma=0.01)
        self.W1 = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.W2 = HiddenLayer((self.nslices, output_shape, 512), rng=rng, gamma=0.01)
    
        self.b0 = BiasLayer((self.nslices, 512))
        self.b1 = BiasLayer((self.nslices, 512))
        self.b2 = BiasLayer((self.nslices, output_shape))

        self.layers = [
            self.W0, self.W1, self.W2,
            self.b0, self.b1, self.b2]

        self.params = sum([layer.params for layer in self.layers], [])
        
    def __call__(self, input):
        
        pscale = self.nslices * input[:,-1]
        pamount = pscale % 1.0
        
        pindex_1 = T.cast(pscale, 'int32') % self.nslices
        pindex_0 = (pindex_1-1) % self.nslices
        pindex_2 = (pindex_1+1) % self.nslices
        pindex_3 = (pindex_1+2) % self.nslices
        
        Wamount = pamount.dimshuffle(0, 'x', 'x')
        bamount = pamount.dimshuffle(0, 'x')
        
        def cubic(y0, y1, y2, y3, mu):
            return (
                (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
                (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
                (-0.5*y0+0.5*y2)*mu +
                (y1))
        
        W0 = cubic(self.W0.W[pindex_0], self.W0.W[pindex_1], self.W0.W[pindex_2], self.W0.W[pindex_3], Wamount)
        W1 = cubic(self.W1.W[pindex_0], self.W1.W[pindex_1], self.W1.W[pindex_2], self.W1.W[pindex_3], Wamount)
        W2 = cubic(self.W2.W[pindex_0], self.W2.W[pindex_1], self.W2.W[pindex_2], self.W2.W[pindex_3], Wamount)
        
        b0 = cubic(self.b0.b[pindex_0], self.b0.b[pindex_1], self.b0.b[pindex_2], self.b0.b[pindex_3], bamount)
        b1 = cubic(self.b1.b[pindex_0], self.b1.b[pindex_1], self.b1.b[pindex_2], self.b1.b[pindex_3], bamount)
        b2 = cubic(self.b2.b[pindex_0], self.b2.b[pindex_1], self.b2.b[pindex_2], self.b2.b[pindex_3], bamount)
        
        H0 = input[:,:-1]
        H1 = self.activation(T.batched_dot(W0, self.dropout0(H0)) + b0)
        H2 = self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1)
        H3 =                 T.batched_dot(W2, self.dropout2(H2)) + b2
        
        return H3
        
    def cost(self, input):
        input = input[:,:-1]
        costs = 0
        for layer in self.layers:
            costs += layer.cost(input)
            input = layer(input)
        return costs / len(self.layers)
    
    def save(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.save(database, '%sL%03i_' % (prefix, li))
        
    def load(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.load(database, '%sL%03i_' % (prefix, li))

            
""" Function to Save Network Weights """

def save_network(network):

    """ Load Control Points """

    W0n = network.W0.W.get_value()
    W1n = network.W1.W.get_value()
    W2n = network.W2.W.get_value()

    b0n = network.b0.b.get_value()
    b1n = network.b1.b.get_value()
    b2n = network.b2.b.get_value()
    
    """ Precompute Phase Function """
    
    for i in range(50):
        
        pscale = network.nslices*(float(i)/50)
        pamount = pscale % 1.0
        
        pindex_1 = int(pscale) % network.nslices
        pindex_0 = (pindex_1-1) % network.nslices
        pindex_2 = (pindex_1+1) % network.nslices
        pindex_3 = (pindex_1+2) % network.nslices
        
        def cubic(y0, y1, y2, y3, mu):
            return (
                (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
                (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
                (-0.5*y0+0.5*y2)*mu +
                (y1))
        
        W0 = cubic(W0n[pindex_0], W0n[pindex_1], W0n[pindex_2], W0n[pindex_3], pamount)
        W1 = cubic(W1n[pindex_0], W1n[pindex_1], W1n[pindex_2], W1n[pindex_3], pamount)
        W2 = cubic(W2n[pindex_0], W2n[pindex_1], W2n[pindex_2], W2n[pindex_3], pamount)
        
        b0 = cubic(b0n[pindex_0], b0n[pindex_1], b0n[pindex_2], b0n[pindex_3], pamount)
        b1 = cubic(b1n[pindex_0], b1n[pindex_1], b1n[pindex_2], b1n[pindex_3], pamount)
        b2 = cubic(b2n[pindex_0], b2n[pindex_1], b2n[pindex_2], b2n[pindex_3], pamount)
        
        W0.astype(np.float32).tofile('./demo/network/pfnn/W0_%03i.bin' % i)
        W1.astype(np.float32).tofile('./demo/network/pfnn/W1_%03i.bin' % i)
        W2.astype(np.float32).tofile('./demo/network/pfnn/W2_%03i.bin' % i)
        
        b0.astype(np.float32).tofile('./demo/network/pfnn/b0_%03i.bin' % i)
        b1.astype(np.float32).tofile('./demo/network/pfnn/b1_%03i.bin' % i)
        b2.astype(np.float32).tofile('./demo/network/pfnn/b2_%03i.bin' % i)

        
""" Construct Network """

network = PhaseFunctionedNetwork(rng=rng, input_shape=X.shape[1]+1, output_shape=Y.shape[1], dropout=0.7)

""" Construct Trainer """

batchsize = 32
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=1, alpha=0.0001)

""" Start Training """

I = np.arange(len(X))

for me in range(20):
    rng.shuffle(I)
    
    print('\n[MacroEpoch] %03i' % me)
    
    for bi in range(10):
    
        """ Find Batch Range """
        
        start, stop = ((bi+0)*len(I))//10, ((bi+1)*len(I))//10
        
        """ Load Data to GPU and train """
        
        E = theano.shared(np.concatenate([X[I[start:stop]], P[I[start:stop]][...,np.newaxis]], axis=-1), borrow=True)
        F = theano.shared(Y[I[start:stop]], borrow=True)
        trainer.train(network, E, F, filename='./demo/network/pfnn/network.npz', restart=False, shuffle=False)
        
        """ Unload Data from GPU """
        
        E.set_value([[]]); del E
        F.set_value([[]]); del F
        
        """ Save Network """
        
        save_network(network)

