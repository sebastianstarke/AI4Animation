"""
Class of ExpertWeights
"""

import numpy as np
import tensorflow as tf

class ExpertWeights(object):
    def __init__(self, rng, shape , name):
        """rng"""
        self.initialRNG   = rng
        
        """shape"""
        self.weight_shape =  shape                    #4/8 * out * in
        self.bias_shape   =  (shape[0],shape[1],1)    #4/8 * out * 1
        
        """alpha and beta"""
        self.alpha        =  tf.Variable(self.initial_alpha(), name=name+'alpha') 
        self.beta         =  tf.Variable(self.initial_beta(),  name=name+'beta') 
    
        
    """initialize parameters for experts i.e. alpha and beta"""
    def initial_alpha_np(self):
        shape = self.weight_shape
        rng   = self.initialRNG
        alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
        alpha = np.asarray(
            rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32)
        return alpha
    
    def initial_alpha(self):
        alpha = self.initial_alpha_np()
        return tf.convert_to_tensor(alpha, dtype = tf.float32)
    
    def initial_beta(self):
        return tf.zeros(self.bias_shape, tf.float32)
    
    def get_NNweight(self, controlweights, batch_size):  
        a = tf.expand_dims(self.alpha, 1)                           #4*out*in   -> 4*1*out*in
        a = tf.tile(a, [1,batch_size,1,1])                          #4*1*out*in -> 4*?*out*in
        w = tf.expand_dims(tf.expand_dims(controlweights, -1), -1)  #4*?        -> 4*?*1*1   
        r = w * a                                                   #4*?*1*1 m 4*?*out*in
        return tf.reduce_sum(r , axis = 0)                          #?*out*in
        
        
    def get_NNbias(self, controlweights, batch_size):
        b = tf.expand_dims(self.beta, 1)                            #4*out*1   -> 4*1*out*1
        b = tf.tile(b, [1,batch_size,1,1])                          #4*1*out*1 -> 4*?*out*1
        w = tf.expand_dims(tf.expand_dims(controlweights, -1), -1)  #4*?        -> 4*?*1*1  
        r = w * b                                                   #4*?*1*1 m 4*?*out*1
        return tf.reduce_sum(r , axis = 0)                          #?*out*1



def save_EP(alpha, beta, filename, num_experts):
    for i in range(len(alpha)):
        for j in range(num_experts):
            a = alpha[i][j]
            b = beta[i][j]
            a.tofile(filename+'/cp%0i_a%0i.bin' % (i,j))
            b.tofile(filename+'/cp%0i_b%0i.bin' % (i,j))



    
    
    
    
"""
def regularization_penalty(alpha, gamma):
    number_alpha = len(alpha)
    penalty = 0
    for i in range(number_alpha):
        penalty += tf.reduce_mean(tf.abs(alpha[i]))
    return gamma * penalty / number_alpha
"""

