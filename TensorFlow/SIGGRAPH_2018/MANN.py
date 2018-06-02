import numpy as np
import tensorflow as tf
import Gating as GT
from Gating import Gating
import ExpertWeights as EW
from ExpertWeights import ExpertWeights
from AdamWParameter import AdamWParameter
from AdamW import AdamOptimizer
import Utils as utils


class MANN(object):
    def __init__(self, 
                 num_joints,
                 num_styles,
                 rng,
                 sess,
                 datapath, savepath,
                 num_experts,
                 hidden_size = 512,
                 hidden_size_gt = 32, 
                 feetJoints = [10, 15, 19, 23],
                 batch_size = 32 , epoch = 150, Te = 10, Tmult =2, 
                 learning_rate_ini = 0.0001, weightDecay_ini = 0.0025, keep_prob_ini = 0.7):
        
        self.num_joints = num_joints
        self.num_styles = num_styles
        self.rng       = rng
        self.sess      = sess
        
        #load data
        self.savepath    = savepath
        utils.build_path([savepath+'/normalization'])
        self.input_data  = utils.Normalize(np.float32(np.loadtxt(datapath+'/Input.txt')), axis = 0, savefile=savepath+'/normalization/X')
        self.output_data = utils.Normalize(np.float32(np.loadtxt(datapath+'/Output.txt')), axis = 0, savefile=savepath+'/normalization/Y')
        self.input_size  = self.input_data.shape[1]
        self.output_size = self.output_data.shape[1]
        self.size_data   = self.input_data.shape[0]
        self.hidden_size = hidden_size
        
        #gatingNN
        self.num_experts    = num_experts
        self.hidden_size_gt = hidden_size_gt
        self.feetJoints     = feetJoints
        
        #training hyperpara
        self.batch_size    = batch_size
        self.epoch         = epoch
        self.total_batch   = int(self.size_data / self.batch_size)
        
        #adamWR controllers
        self.AP = AdamWParameter(nEpochs      = self.epoch,
                                 Te           = Te,
                                 Tmult        = Tmult,
                                 LR           = learning_rate_ini, 
                                 weightDecay  = weightDecay_ini,
                                 batchSize    = self.batch_size,
                                 nBatches     = self.total_batch
                                 )
        #keep_prob
        self.keep_prob_ini     = keep_prob_ini
        
        
        
        
    def build_model(self):
        #Placeholders
        self.nn_X         = tf.placeholder(tf.float32, [self.batch_size, self.input_size],  name='nn_X') 
        self.nn_Y         = tf.placeholder(tf.float32, [self.batch_size, self.output_size], name='nn_Y')  
        self.nn_keep_prob = tf.placeholder(tf.float32, name = 'nn_keep_prob') 
        self.nn_lr_c      = tf.placeholder(tf.float32, name = 'nn_lr_c') 
        self.nn_wd_c      = tf.placeholder(tf.float32, name = 'nn_wd_c')
        
        """BUILD gatingNN"""
        #input of gatingNN
        self.gating_input, self.input_size_gt = GT.getInput(self.nn_X, self.feetJoints, self.num_styles)
        self.gating_input = tf.transpose(self.gating_input)
        self.gatingNN = Gating(self.rng, self.gating_input, self.input_size_gt, self.num_experts, self.hidden_size_gt, self.nn_keep_prob)
        #bleding coefficients
        self.BC = self.gatingNN.BC
        
        #initialize experts
        self.layer0 = ExpertWeights(self.rng, (self.num_experts, self.hidden_size,  self.input_size),   'layer0') # alpha: 4/8*hid*in, beta: 4/8*hid*1
        self.layer1 = ExpertWeights(self.rng, (self.num_experts, self.hidden_size, self.hidden_size),   'layer1') # alpha: 4/8*hid*hid,beta: 4/8*hid*1
        self.layer2 = ExpertWeights(self.rng, (self.num_experts, self.output_size, self.hidden_size),   'layer2') # alpha: 4/8*out*hid,beta: 4/8*out*1 
        
        
        #initialize parameters in main NN
        """
        dimension of w: ?* out* in
        dimension of b: ?* out* 1
        """
        w0  = self.layer0.get_NNweight(self.BC, self.batch_size)
        w1  = self.layer1.get_NNweight(self.BC, self.batch_size)
        w2  = self.layer2.get_NNweight(self.BC, self.batch_size)
        
        b0  = self.layer0.get_NNbias(self.BC, self.batch_size)
        b1  = self.layer1.get_NNbias(self.BC, self.batch_size)
        b2  = self.layer2.get_NNbias(self.BC, self.batch_size)
        
        #build main NN
        H0 = tf.expand_dims(self.nn_X, -1)                     #?*in -> ?*in*1
        H0 = tf.nn.dropout(H0, keep_prob=self.nn_keep_prob)        
        
        H1 = tf.matmul(w0, H0) + b0                            #?*out*in mul ?*in*1 + ?*out*1 = ?*out*1
        H1 = tf.nn.elu(H1)             
        H1 = tf.nn.dropout(H1, keep_prob=self.nn_keep_prob) 
        
        H2 = tf.matmul(w1, H1) + b1
        H2 = tf.nn.elu(H2)             
        H2 = tf.nn.dropout(H2, keep_prob=self.nn_keep_prob) 
        
        H3 = tf.matmul(w2, H2) + b2
        self.H3 = tf.squeeze(H3, -1)                           #?*out*1 ->?*out  
        
        self.loss       = tf.reduce_mean(tf.square(self.nn_Y - self.H3))
        self.optimizer  = AdamOptimizer(learning_rate= self.nn_lr_c, wdc =self.nn_wd_c).minimize(self.loss)
        
                
                

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        """training"""
        print("total_batch:", self.total_batch)
        #randomly select training set
        I = np.arange(self.size_data)
        self.rng.shuffle(I)
        error_train = np.ones(self.epoch)
        #saving path
        model_path   = self.savepath+ '/model'
        nn_path      = self.savepath+ '/nn'
        weights_path = self.savepath+ '/weights'
        utils.build_path([model_path, nn_path, weights_path])
        
        #start to train
        print('Learning starts..')
        for epoch in range(self.epoch):
            avg_cost_train = 0
            for i in range(self.total_batch):
                index_train = I[i*self.batch_size:(i+1)*self.batch_size]
                batch_xs = self.input_data[index_train]
                batch_ys = self.output_data[index_train]
                clr, wdc = self.AP.getParameter(epoch)   #currentLearningRate & weightDecayCurrent
                feed_dict = {self.nn_X: batch_xs, self.nn_Y: batch_ys, self.nn_keep_prob: self.keep_prob_ini, self.nn_lr_c: clr, self.nn_wd_c: wdc}
                l, _, = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
                avg_cost_train += l / self.total_batch
                
                if i % 1000 == 0:
                    print(i, "trainingloss:", l)
                    print('Epoch:', '%04d' % (epoch + 1), 'clr:', clr)
                    print('Epoch:', '%04d' % (epoch + 1), 'wdc:', wdc)
                    
            #print and save training test error 
            print('Epoch:', '%04d' % (epoch + 1), 'trainingloss =', '{:.9f}'.format(avg_cost_train))
            
            error_train[epoch] = avg_cost_train
            error_train.tofile(model_path+"/error_train.bin")

            #save model and weights
            saver.save(self.sess, model_path+"/model.ckpt")
            GT.save_GT((self.sess.run(self.gatingNN.w0), self.sess.run(self.gatingNN.w1), self.sess.run(self.gatingNN.w2)), 
                       (self.sess.run(self.gatingNN.b0), self.sess.run(self.gatingNN.b1), self.sess.run(self.gatingNN.b2)), 
                       nn_path
                       )
            EW.save_EP((self.sess.run(self.layer0.alpha), self.sess.run(self.layer1.alpha), self.sess.run(self.layer2.alpha)),
                       (self.sess.run(self.layer0.beta), self.sess.run(self.layer1.beta), self.sess.run(self.layer2.beta)),
                       nn_path,
                       self.num_experts
                       )
            
            if epoch%10==0:
                weights_nn_path = weights_path + '/nn%03i' % epoch
                utils.build_path([weights_nn_path])
                GT.save_GT((self.sess.run(self.gatingNN.w0), self.sess.run(self.gatingNN.w1), self.sess.run(self.gatingNN.w2)), 
                           (self.sess.run(self.gatingNN.b0), self.sess.run(self.gatingNN.b1), self.sess.run(self.gatingNN.b2)), 
                           weights_nn_path
                           )
                EW.save_EP((self.sess.run(self.layer0.alpha), self.sess.run(self.layer1.alpha), self.sess.run(self.layer2.alpha)),
                           (self.sess.run(self.layer0.beta), self.sess.run(self.layer1.beta), self.sess.run(self.layer2.beta)),
                           weights_nn_path,
                           self.num_experts
                           )
        print('Learning Finished')