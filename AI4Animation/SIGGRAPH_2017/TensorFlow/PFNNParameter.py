import numpy as np
import tensorflow as tf

tf.set_random_seed(23456)  # reproducibility

class PFNNParameter:
    def __init__(self, shape, rng, phase, name):
        
        """shape"""
        self.control_num  = shape[0]  #4 control points
        self.weight_shape = shape
        self.bias_shape   = shape[:-1]
        
        """"alpha and beta"""
        self.alpha        =  tf.Variable(self.initial_alpha(rng), name=name+'alpha') 
        self.beta         =  tf.Variable(self.initial_beta(), name=name+'beta') 
        
        """index and weight for phase function"""
        self.pindex_1, self.bamount, self.wamount = self.getIndexAmount(phase)
        self.pindex_0     = (self.pindex_1-1) % self.control_num
        self.pindex_2     = (self.pindex_1+1) % self.control_num
        self.pindex_3     = (self.pindex_1+2) % self.control_num
        
        """weight and bias"""
        self.weight       = self.cotrol(1)
        self.bias         = self.cotrol(0)
                            
        
    """initialize parameters in phase function i.e. alpha and beta"""
    def initial_alpha(self, rng):
        shape = self.weight_shape
        alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
        alpha = np.asarray(
            rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32)
        return tf.convert_to_tensor(alpha, dtype = tf.float32)
    
    def initial_beta(self):
        return tf.zeros(self.bias_shape, tf.float32)
    
    """ cubic function to calculate the weights and bias in nn"""
    
    def cotrol(self, flag):
        if flag:
            y0 = tf.nn.embedding_lookup(self.alpha, self.pindex_0)
            y1 = tf.nn.embedding_lookup(self.alpha, self.pindex_1)
            y2 = tf.nn.embedding_lookup(self.alpha, self.pindex_2)
            y3 = tf.nn.embedding_lookup(self.alpha, self.pindex_3)
            mu = self.wamount
        else:
            y0 = tf.nn.embedding_lookup(self.beta, self.pindex_0)
            y1 = tf.nn.embedding_lookup(self.beta, self.pindex_1)
            y2 = tf.nn.embedding_lookup(self.beta, self.pindex_2)
            y3 = tf.nn.embedding_lookup(self.beta, self.pindex_3)
            mu = self.bamount
        return cubic(y0, y1, y2, y3, mu)

    def getIndexAmount(self, phase):
        nslices = self.control_num                    # number of control points in phase function
        pscale = nslices * phase
        pamount = pscale % 1.0
        pindex_1 = tf.cast(pscale, 'int32') % nslices
        
        bamount = tf.expand_dims(pamount, 1)
        wamount = tf.expand_dims(bamount, 1)          # expand 2 dimension [n*1*1]
        return  pindex_1, bamount, wamount
    

def cubic(y0, y1, y2, y3, mu):
    return (
        (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
        (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
        (-0.5*y0+0.5*y2)*mu +
        (y1))



"""save 50 weights and bias for precomputation in real-time """
def save_network(alpha, beta, num_points,filename):
    nslices = 4
    for i in range(num_points):
        """calculate the index and weights in phase function """
        pscale = nslices*(float(i)/50)
        #weight
        pamount = pscale % 1.0
        #index
        pindex_1 = int(pscale) % nslices
        pindex_0 = (pindex_1-1) % nslices
        pindex_2 = (pindex_1+1) % nslices
        pindex_3 = (pindex_1+2) % nslices
        
        for j in range(len(alpha)):
            a = alpha[j]
            b = beta[j]
            W = cubic(a[pindex_0],a[pindex_1],a[pindex_2],a[pindex_3],pamount)
            B = cubic(b[pindex_0],b[pindex_1],b[pindex_2],b[pindex_3],pamount)

            W.tofile(filename+'/W%0i_%03i.bin' % (j,i))
            B.tofile(filename+'/b%0i_%03i.bin' % (j,i))



def regularization_penalty(alpha, gamma):
    number_alpha = len(alpha)
    penalty = 0
    for i in range(number_alpha):
        penalty += tf.reduce_mean(tf.abs(alpha[i]))
    return gamma * penalty / number_alpha




