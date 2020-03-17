import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys

sys.path.append('../Lib_Expert')
sys.path.append('../Lib_Optimizer')
sys.path.append('../Lib_Opt')
from ComponentNN import ComponentNN
from AdamWParameter import AdamWParameter
from AdamW import AdamOptimizer
from NeuralNetwork import NeuralNetwork
from MLP import MLP


class MainNN(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self)

    def BuildModel(self,
                   # For Data Process
                   load_path, save_path, type_normalize,

                   # For Network Structure
                   expert_components,  # Number of Experts for each NN
                   input_components,  # Index of Inputs for each NN
                   dim_components,  # Dims of Hidden layers for each NN
                   act_components,  # Activation function for each layers each NN

                   # For encoder
                   index_encoders=[],
                   dim_encoders=[],
                   activation_encoders=[],
                   ):

        # Process Data, why process here because need the Dims to build Tensors
        self.ProcessData(load_path, save_path, type_normalize)
        # Initialize Placeholders
        self.BuildConstantPlaceHolder()
        self.nn_lr_c = tf.placeholder(tf.float32, name='nn_lr_c')
        self.nn_wd_c = tf.placeholder(tf.float32, name='nn_wd_c')

        # Encoder
        self.num_encoders = len(index_encoders)
        if self.num_encoders > 0:
            self.encoders = []
            input_finalComp = []
            for i in range(self.num_encoders):
                encoder_input = tf.gather(self.nn_X, index_encoders[i], axis=-1)
                encoder_dims = [len(index_encoders[i])] + dim_encoders[i]
                encoder_activations = activation_encoders[i]
                encoder_keep_prob = self.nn_keep_prob
                encoder = MLP(self.rng, encoder_input, encoder_dims, encoder_activations, encoder_keep_prob,
                              'encoder%0i' % i)
                self.encoders.append(encoder)
                # SAVER FOR ENCODERS
                tf.add_to_collection("nn_latent_encoder%0i" % i, encoder.output)
                input_finalComp.append(encoder.output)
            input_finalComp = tf.concat(input_finalComp, axis=-1)
        else:
            input_finalComp = tf.gather(self.nn_X, input_components[-1], axis=1)
        # PLOT ENCODERS
        for i in range(self.num_encoders):
            print('--------------------')
            print('Encoder', i + 1)
            print('Pivot', index_encoders[i][0])
            for j in range(len(self.encoders[i].dim_layers)):
                print('Layer', j + 1, self.encoders[i].dim_layers[j], self.encoders[i].activation)

        # Gating Network
        num_components = len(expert_components)
        weight_blend = tf.ones((expert_components[0], self.nn_batch_size), dtype=tf.float32)
        comp_first = ComponentNN(self.rng,
                                 tf.gather(self.nn_X, input_components[0], axis=1),
                                 expert_components[0],
                                 [len(input_components[0])] + dim_components[0] + [expert_components[1]],
                                 act_components[0], weight_blend, self.nn_keep_prob, self.nn_batch_size,
                                 'comp0')
        weight_blend = tf.transpose(comp_first.output)

        self.comps = [comp_first]
        for i in range(num_components - 2):
            comp = ComponentNN(self.rng, tf.gather(self.nn_X, input_components[i + 1], axis=1),
                               expert_components[i + 1],
                               [len(input_components[i + 1])] + dim_components[i + 1] + [
                                   expert_components[i + 2]], act_components[i + 1], weight_blend,
                               self.nn_keep_prob, self.nn_batch_size, 'comp%i' % (i + 1))
            weight_blend = tf.transpose(comp.output)
            self.comps.append(comp)
        # GATING OUTPUT AND SAVER
        tf.add_to_collection("nn_blend_weights", tf.transpose(weight_blend))

        # Motion Network
        comp_final = ComponentNN(self.rng, input_finalComp,
                                 expert_components[num_components - 1],
                                 [input_finalComp.shape[-1].value] + dim_components[-1] + [self.output_dim],
                                 act_components[num_components - 1], weight_blend,
                                 self.nn_keep_prob, self.nn_batch_size,
                                 'comp%i' % (num_components - 1))
        self.comps.append(comp_final)
        # SAVER
        tf.add_to_collection("nn_prediction", comp_final.output)

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.nn_Y - comp_final.output))
        self.optimizer = AdamOptimizer(learning_rate=self.nn_lr_c, wdc=self.nn_wd_c).minimize(self.loss)

    def Train(self,
              # Tensorflow Session
              sess,
              # Model name for Saving
              name_model,
              # Flag/Path of test data
              path_test='',
              # Flag for saving model and Flag for saving bin
              flag_save_tf=True, flag_save_bin=False,
              # Saving settings
              step_save=1,
              max_saving=1,
              # HyperParameters
              keep_prob=0.7,
              batch_size=32, epoch=150, Te=10, Tmult=2,
              learning_rate_ini=0.0001, weightDecay_ini=0.0025
              ):

        # Print Training Information
        total_batch = int(self.data_size / batch_size)
        print('Training information')
        print('Total Batch', '->', total_batch)
        print('--------------------')
        print('Input', 'X', '->', self.input_dim)
        print('Output', 'Y', '->', self.output_dim)
        for i in range(len(self.comps)):
            print('--------------------')
            print('Network', i + 1)
            for j in range(len(self.comps[i].dim_layers)):
                print('Layer', j + 1, self.comps[i].dim_layers[j], self.comps[i].activation)

        # Print Testing Information if test
        if (len(path_test) > 0):
            self.LoadTestData(path_test)
            if self.data_size_test > batch_size:
                batch_size_test = batch_size
            else:
                batch_size_test = self.data_size_test
            total_batch_test = int(self.data_size_test / batch_size_test)
        # Initialize Graph
        sess.run(tf.global_variables_initializer())
        # Initialize Saver
        saver = tf.train.Saver(max_to_keep=max_saving)
        # Start training
        self.AP = AdamWParameter(nEpochs=epoch,
                                 Te=Te,
                                 Tmult=Tmult,
                                 LR=learning_rate_ini,
                                 weightDecay=weightDecay_ini,
                                 batchSize=batch_size,
                                 nBatches=total_batch
                                 )
        I = np.arange(self.data_size)
        train_loss = []
        test_loss = []
        for e in range(epoch):
            # Randomly select training set
            self.rng.shuffle(I)
            avg_cost_train = 0
            for i in range(total_batch):
                print('Progress', round(i / total_batch, 2), "%", end="\r")
                index_train = I[i * batch_size:(i + 1) * batch_size]
                batch_xs = self.input_data[index_train]
                batch_ys = self.output_data[index_train]
                clr, wdc = self.AP.getParameter(e)  # currentLearningRate and weightDecayCurrent
                feed_dict = {self.nn_batch_size: batch_size,
                             self.nn_X: batch_xs,
                             self.nn_Y: batch_ys,
                             self.nn_lr_c: clr,
                             self.nn_wd_c: wdc,
                             self.nn_keep_prob: keep_prob}
                l, _, = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
                avg_cost_train += l / total_batch
            print('Epoch:', '%04d' % (e + 1), 'Training Loss =', '{:.9f}'.format(avg_cost_train))
            train_loss.append(avg_cost_train)

            # Test
            if (len(path_test) > 0):
                avg_cost_test = 0
                for i in range(total_batch_test):
                    batch_xs = self.input_test[i * batch_size_test:(i + 1) * batch_size_test]
                    batch_ys = self.output_test[i * batch_size_test:(i + 1) * batch_size_test]
                    feed_dict = {self.nn_batch_size: batch_size_test,
                                 self.nn_X: batch_xs,
                                 self.nn_Y: batch_ys,
                                 self.nn_keep_prob: 1.0}
                    l = sess.run(self.loss, feed_dict=feed_dict)
                    avg_cost_test += l / total_batch_test
                print('Epoch:', '%04d' % (e + 1), 'Testing Loss =', '{:.9f}'.format(avg_cost_test))
                test_loss.append(avg_cost_test)

            # SAVER
            if flag_save_bin:
                for i in range(len(self.comps)):
                    self.comps[i].saveNN(sess, self.save_path, i)
                if self.num_encoders > 0:
                    for i in range(self.num_encoders):
                        self.encoders[i].saveNN(sess, self.save_path + '/encoder%0i' % i)

            if flag_save_tf and e % step_save == 0:
                saver.save(sess, self.save_path + '/' + name_model, global_step=e)
        np.array(train_loss, dtype=np.float32).tofile(self.save_path + '/' + 'trainloss.bin')
        np.array(test_loss, dtype=np.float32).tofile(self.save_path + '/' + 'testloss.bin')
        print('Learning Finished')
