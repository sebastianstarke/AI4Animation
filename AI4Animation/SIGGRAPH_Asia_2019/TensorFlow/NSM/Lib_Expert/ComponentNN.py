"""
Class of GatingNN or ComponentNN
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from ExpertWeights import ExpertWeights


class ComponentNN(object):
    def __init__(self, rng, input_x, num_experts, dim_layers, activation, weight_blend, keep_prob, batchSize, name,
                 FiLM=None):
        """

        :param rng: random seed for numpy
        :param input_x: input tensor of ComponentNN/GatingNN
        :param num_experts: number of experts
        :param dim_layers: dimension of each layer including the dimension of input and output
        :param activation: activation function of each layer
        :param weight_blend: blending weights from previous ComponentNN that used experts in
                             current Components NN.
                             Note that the VanillaNN can also be represented as Components NN with 1 Expert
        :param keep_prob: for drop out
        :param batchSize: for batch size
        :param name: for name of current component
        :param FiLM: Technique of FiLM, will not use this one in default
        """
        self.name = name

        """rng"""
        self.initialRNG = rng

        """input"""
        self.input = input_x

        """dropout"""
        self.keep_prob = keep_prob

        """"NN structure"""
        self.num_experts = num_experts
        self.dim_layers = dim_layers
        self.num_layers = len(dim_layers) - 1
        self.activation = activation
        self.batchSize = batchSize

        """Build NN"""
        self.experts = self.initExperts()
        self.network = self.buildNN(weight_blend, FiLM)

    def initExperts(self):
        experts = []
        for i in range(self.num_layers):
            expert = ExpertWeights(self.initialRNG, (self.num_experts, self.dim_layers[i + 1], self.dim_layers[i]),
                                   self.name + 'layer%i' % i)
            experts.append(expert)
        return experts

    def buildNN(self, weight_blend, FiLM):
        H = tf.expand_dims(self.input, -1)  # ?*in -> ?*in*1
        H = tf.nn.dropout(H, keep_prob=self.keep_prob)
        for i in range(self.num_layers - 1):
            w = self.experts[i].get_NNweight(weight_blend, self.batchSize)
            b = self.experts[i].get_NNbias(weight_blend, self.batchSize)

            H = tf.matmul(w, H) + b  # ?*out*in mul ?*in*1 + ?*out*1 = ?*out*1

            if FiLM and i == 0:
                H = tf.squeeze(H, -1)
                H = tf.add(tf.multiply(FiLM[0], H), FiLM[1])
                H = tf.expand_dims(H, -1)
            acti = self.activation[i]
            if acti != 0:
                if acti == tf.nn.softmax:
                    H = acti(H, axis=1)
                else:
                    H = acti(H)
            H = tf.nn.dropout(H, keep_prob=self.keep_prob)

        w = self.experts[self.num_layers - 1].get_NNweight(weight_blend, self.batchSize)
        b = self.experts[self.num_layers - 1].get_NNbias(weight_blend, self.batchSize)
        H = tf.matmul(w, H) + b
        H = tf.squeeze(H, -1)  # ?*out*1 ->?*out
        acti = self.activation[self.num_layers - 1]
        if acti != 0:
            if acti == tf.nn.softmax:
                H = acti(H, axis=1)
            else:
                H = acti(H)
        self.output = H

    def saveNN(self, sess, savepath, index_component):
        """
        :param index_component: index of current component
        """
        for i in range(self.num_layers):
            for j in range(self.num_experts):
                sess.run(self.experts[i].alpha[j]).tofile(savepath + '/wc%0i%0i%0i_w.bin' % (index_component, i, j))
                sess.run(self.experts[i].beta[j]).tofile(savepath + '/wc%0i%0i%0i_b.bin' % (index_component, i, j))
