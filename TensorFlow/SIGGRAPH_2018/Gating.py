"""
Class of Gating NN
"""
import numpy as np
import tensorflow as tf

class Gating(object):
    def __init__(self, rng, input_x, input_size, output_size, hidden_size, keep_prob):
        """rng"""
        self.initialRNG   = rng
        
        """input"""
        self.input        = input_x
        
        """dropout"""
        self.keep_prob    = keep_prob
        
        """size"""
        self.input_size   = input_size
        self.output_size  = output_size
        self.hidden_size  = hidden_size
        
        """parameters"""
        self.w0           = tf.Variable(self.initial_weight([hidden_size, input_size ]),name = 'wc0_w')
        self.w1           = tf.Variable(self.initial_weight([hidden_size, hidden_size]),name = 'wc1_w')
        self.w2           = tf.Variable(self.initial_weight([output_size, hidden_size]),name = 'wc2_w')
        
        self.b0           = tf.Variable(self.initial_bias([hidden_size, 1]) ,name = 'wc0_b')
        self.b1           = tf.Variable(self.initial_bias([hidden_size, 1]) ,name = 'wc1_b')
        self.b2           = tf.Variable(self.initial_bias([output_size, 1]) ,name = 'wc2_b')
        
        """"output blending coefficients"""
        self.BC   = self.fp()
        
        
    """initialize parameters """
    def initial_weight(self, shape):
        rng   = self.initialRNG
        weight_bound = np.sqrt(6. / np.sum(shape[-2:]))
        weight = np.asarray(
            rng.uniform(low=-weight_bound, high=weight_bound, size=shape),
            dtype=np.float32)
        return tf.convert_to_tensor(weight, dtype = tf.float32)
    
    def initial_bias(self, shape):
        return tf.zeros(shape, tf.float32)
    
    
    """forward propogation"""
    def fp(self):
        H0 = tf.nn.dropout(self.input, keep_prob=self.keep_prob) #input*batch
        
        H1 = tf.matmul(self.w0, H0) + self.b0                    #hidden*input mul input*batch       
        H1 = tf.nn.elu(H1)             
        H1 = tf.nn.dropout(H1, keep_prob=self.keep_prob) 
        
        H2 = tf.matmul(self.w1, H1) + self.b1     
        H2 = tf.nn.elu(H2)             
        H2 = tf.nn.dropout(H2, keep_prob=self.keep_prob) 
        
        
        H3 = tf.matmul(self.w2, H2) + self.b2                    #out*hidden   mul hidden*batch
        H3 = tf.nn.softmax(H3,dim = 0)                           #out*batch
        return H3



#--------------------------------------get the input for the Gating network---------------------------------
"""global parameters"""
num_trajPoints       = 12 #number of trajectory points
num_trajUnit_noSpeed = 6  #number of trajectory units: Position X,Z; Direction X,Z; Velocity X,Z;           
num_trajUnit_speed   = 7  #number of trajectory units: Position X,Z; Direction X,Z; Velocity X,Z; Speed
num_jointUnit        = 12 #number of joint units: PositionXYZ Rotation VelocityXYZ


#get the velocity of joints, desired velocity and style
def getInput(data, index_joint):    
    gating_input = data[..., index_joint[0]:index_joint[0]+1]
    index_joint.remove(index_joint[0])
    for i in index_joint:
        gating_input  = tf.concat( [gating_input, data[...,i:i+1]],axis = -1)
    return gating_input 


def save_GT(weight, bias, filename):
    for i in range(len(weight)):
        a = weight[i]
        b = bias[i]
        a.tofile(filename+'/wc%0i_w.bin' % i)
        b.tofile(filename+'/wc%0i_b.bin' % i)




 
"""
def regularization_penalty(weight, gamma):
    number_weight = len(weight)
    penalty = 0
    for i in range(number_weight):
        penalty += tf.reduce_mean(tf.abs(weight[i]))
    return gamma * penalty / number_weight
"""







