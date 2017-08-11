import numpy as np
import theano
import theano.tensor as T

from Layer import Layer

class ActivationLayer(Layer):

    def __init__(self, f='ReLU', params=[]):
        
        if   f == 'ReLU':
            self.f = lambda x: T.switch(x<0,0,x)
        elif f == 'LReLU':
            self.f = lambda x: T.switch(x<0,0.01*x,x)
        elif f == 'ELU':
            self.f = lambda x: T.switch(x<0, (T.exp(x) - 1), x)
        elif f == 'softplus':
            self.f = lambda x: T.log(1 + T.exp(x))
        elif f == 'tanh':
            self.f = T.tanh
        elif f == 'sigmoid':
            self.f = T.nnet.sigmoid
        elif f == 'identity':
            self.f = lambda x: x
        else:
            self.f = f
        
        self.params = params
        
    def __call__(self, input): return self.f(input)
        