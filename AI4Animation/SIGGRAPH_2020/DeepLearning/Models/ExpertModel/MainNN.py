import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from ComponentNN import ComponentNN
import ComponentNN as COMP
import ExpertWeights as EW
from ExpertWeights import ExpertWeights
from AdamWParameter import AdamWParameter
from AdamW import AdamOptimizer
import Utils as utils
from itertools import islice

class MainNN(object):
    def __init__(self, 
                 rng,
                 sess,
                 
                 file_input, 
                 file_output, 
                 norm_input, 
                 norm_output, 
                 savepath,

                 expert_components,
                 input_components,
                 output_components,
                 dim_components,
                 act_components,
                 keep_prob_components, 
                                  
                 batch_size = 32 , epoch = 150, Te = 10, Tmult = 2, 
                 learning_rate_ini = 0.0001, weightDecay_ini = 0.0025
                ):
        
        self.file_input = file_input
        self.file_output = file_output
        self.rng       = rng
        self.sess      = sess
        
        #load data
        self.savepath    = savepath

        print("Started creating data pointers...")
        self.pointersX = np.array(utils.CollectPointers(file_input))
        self.pointersY = np.array(utils.CollectPointers(file_output))
        print("Finished creating data pointers.")

        print("Reading normalization values.")
        self.input_norm = utils.ReadNorm(norm_input)
        self.output_norm = utils.ReadNorm(norm_output)
        if not output_components == [Ellipsis]:
            self.output_data = self.output_data[:,output_components]
            self.output_norm = self.output_norm[:,output_components]

        self.input_size  = self.input_norm.shape[1]
        self.output_size = self.output_norm.shape[1]
        if self.pointersX.shape != self.pointersY.shape:
            print("Mismatching input sizes")
            return
        self.size_data   = self.pointersX.shape[0]
        
        #Components NN
        self.num_components         = len(expert_components)
        self.expert_components      = expert_components
        self.input_components       = input_components
        self.dim_components         = dim_components
        self.act_components         = act_components
        self.keep_prob_components   = keep_prob_components
        
        #training hyperparameters
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

    def build_model(self):
        #Placeholders
        self.nn_X         = tf.placeholder(tf.float32, [self.batch_size, self.input_size],  name='nn_X') 
        self.nn_Y         = tf.placeholder(tf.float32, [self.batch_size, self.output_size], name='nn_Y')  
        self.nn_lr_c      = tf.placeholder(tf.float32, name = 'nn_lr_c') 
        self.nn_wd_c      = tf.placeholder(tf.float32, name = 'nn_wd_c')
        
        #Gating Network
        weight_blend      = tf.ones((self.expert_components[0], self.batch_size),dtype=tf.float32)
        comp_first        = ComponentNN(self.rng, tf.gather(self.nn_X, self.input_components[0], axis=1), self.expert_components[0], [len(self.input_components[0])] + self.dim_components[0] + [self.expert_components[1]], self.act_components[0], weight_blend, self.keep_prob_components[0], self.batch_size, 'comp0', True)
        weight_blend      = tf.transpose(comp_first.output)
        
        self.comps        = [comp_first]
        for i in range(self.num_components-2):
            comp          = ComponentNN(self.rng, tf.gather(self.nn_X, self.input_components[i+1], axis=1), self.expert_components[i+1], [len(self.input_components[i+1])] + self.dim_components[i+1] + [self.expert_components[i+2]], self.act_components[i+1], weight_blend, self.keep_prob_components[i+1], self.batch_size, 'comp%i'%(i+1), True)
            weight_blend  = tf.transpose(comp.output)
            self.comps.append(comp)
        
        #Motion Network
        comp_final        = ComponentNN(self.rng, tf.gather(self.nn_X, self.input_components[-1], axis=1), self.expert_components[self.num_components-1], [len(self.input_components[-1])] + self.dim_components[-1] + [self.output_size], self.act_components[self.num_components-1], weight_blend, self.keep_prob_components[self.num_components-1], self.batch_size, 'comp%i'%(self.num_components-1), True)
        self.comps.append(comp_final)

        self.loss       = tf.reduce_mean(tf.square(self.nn_Y - comp_final.output))
        self.optimizer  = AdamOptimizer(learning_rate= self.nn_lr_c, wdc =self.nn_wd_c).minimize(self.loss)                
    
    def train(self):
        self.sess.run(tf.global_variables_initializer())
        
        """training"""
        #randomly select training set
        I = np.arange(self.size_data)
        
        #start to train
        print('Learning starts...')
        print('Batch Size', '->', self.total_batch)
        print('--------------------')
        print('Input', 'X', '->', self.input_size)
        print('Output', 'Y', '->', self.output_size)

        for i in range(len(self.comps)):
            print('--------------------')
            print('Network', i+1)
            for j in range(len(self.comps[i].dim_layers)):
                print('Layer', j+1, self.comps[i].dim_layers[j])
            print('Activation', self.comps[i].activation)
        
        for epoch in range(self.epoch):
            self.rng.shuffle(I)
            avg_cost_train = 0.0
            for i in range(self.total_batch):
                print('Progress', round(i/self.total_batch, 2), "%", end="\r")
                index_train = I[i*self.batch_size:(i+1)*self.batch_size]
                # batch_xs = self.input_data[index_train]
                # batch_ys = self.output_data[index_train]
                batch_xs  = utils.ReadChunk(self.file_input, self.pointersX[index_train])
                batch_ys = utils.ReadChunk(self.file_output, self.pointersY[index_train])
                batch_xs = utils.Normalize(batch_xs, self.input_norm)
                batch_ys = utils.Normalize(batch_ys, self.output_norm)
                clr, wdc = self.AP.getParameter(epoch)
                feed_dict = {self.nn_X: batch_xs, self.nn_Y: batch_ys, self.nn_lr_c: clr, self.nn_wd_c: wdc}
                l, www, _, = self.sess.run([self.loss,self.comps[0].output, self.optimizer], feed_dict=feed_dict)
                avg_cost_train += l / self.total_batch
                    
            print('Epoch:', '%04d' % (epoch + 1), 'trainingloss =', '{:.9f}'.format(avg_cost_train))
            
            #save model and weights
            savepath = self.savepath + '/' + str(epoch+1)
            utils.BuildPath([savepath])
            self.input_norm[0].tofile(savepath+'/Xmean.bin')
            self.input_norm[1].tofile(savepath+'/Xstd.bin')
            self.output_norm[0].tofile(savepath+'/Ymean.bin')
            self.output_norm[1].tofile(savepath+'/Ystd.bin')
            for i in range(self.num_components):
                comp    = self.comps[i]
                num_lay = comp.num_layers
                num_ex  = comp.num_experts
                for j in range(num_lay):
                    for k in range(num_ex):
                        self.sess.run(comp.experts[j].alpha[k]).tofile(savepath+'/wc%0i%0i%0i_w.bin' %(i,j,k))
                        self.sess.run(comp.experts[j].beta[k]).tofile(savepath+'/wc%0i%0i%0i_b.bin' %(i,j,k))

        print('Learning Finished')
        
