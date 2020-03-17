# Vanilla MLP
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class MLP(object):
    def __init__(self, rng, input_x, dim_layers, activation, keep_prob, name):
        """
        feed forward network that is usually used as encoder/decoder
        :param rng: random seed for initialization
        :param input_x: input tensor (batch_size * dim)
        :param dim_layers: list of int values for dim
        :param activation: list of activation functions
        :param keep_prob: keep prob
        :param name: name
        """
        self.name = name

        """rng"""
        self.initialRNG = rng

        """input"""
        self.input = input_x

        """dropout"""
        self.keep_prob = keep_prob

        """"NN structure"""
        self.dim_layers = dim_layers
        self.activation = activation
        self.num_layer = len(activation)
        """Build NN"""
        self.buildNN()

    def buildNN(self, initialzer=tf.keras.initializers.glorot_normal):
        assert self.num_layer + 1 == len(self.dim_layers)
        self.W = []
        self.b = []
        for i in range(self.num_layer):
            weights = tf.get_variable(self.name + "W%0i" % i, shape=[self.dim_layers[i + 1], self.dim_layers[i]],
                                      initializer=initialzer)
            bias = tf.Variable(tf.zeros((self.dim_layers[i + 1], 1), tf.float32), name=self.name + 'b%0i' % i)
            self.W.append(weights)
            self.b.append(bias)
        H = tf.transpose(self.input)
        for i in range(self.num_layer):
            H = tf.nn.dropout(H, keep_prob=self.keep_prob)
            H = tf.matmul(self.W[i], H) + self.b[i]
            if self.activation[i] != 0:
                if self.activation[i] == tf.nn.softmax:
                    H = self.activation[i](H, axis=0)
                else:
                    H = self.activation[i](H)
        self.output = tf.transpose(H)

    def saveNN(self, sess, path):
        for i in range(len(self.W)):
            sess.run(self.W[i]).tofile(path + '_w%0i.bin' % i)
            sess.run(self.b[i]).tofile(path + '_b%0i.bin' % i)
