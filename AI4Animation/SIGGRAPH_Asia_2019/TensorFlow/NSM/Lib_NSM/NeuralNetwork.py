# Basic Class for MainNN
import sys
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sys.path.append('../Lib_Utils')
import Utils as utils


class NeuralNetwork(object):
    def __init__(self):
        self.name_tensor_input = "nn_input"
        self.name_tensor_output = "nn_output"
        self.name_tensor_batch_size = "nn_batch_size"
        self.name_tensor_keep_prob = "nn_keep_prob"
        # for reproducibility
        self.rng = np.random.RandomState(1234)
        tf.set_random_seed(1234)

    def ProcessData(self, load_path, save_path, type_normalize=0):
        """
        :param load_path: path of data
        :param save_path: path for saving
        :param type_normalize: how to normalize the data
                0. (default normalization) load mean and std from txt and (data-mean)/std
                1. coming soon

        :return: a. build savepath
                 b. laod data
                 c. get the input dim, output dim and data size
        """
        self.save_path = save_path
        utils.build_path([save_path])
        if (type_normalize == 0):
            self.input_data, self.input_mean, self.input_std = utils.Normalize(
                np.float32(np.loadtxt(load_path + '/Input.txt')),
                np.float32(np.loadtxt(load_path + '/InputNorm.txt')),
                savefile=save_path + '/X')
            self.output_data, self.output_mean, self.output_std = utils.Normalize(
                np.float32(np.loadtxt(load_path + '/Output.txt')),
                np.float32(np.loadtxt(load_path + '/OutputNorm.txt')),
                savefile=save_path + '/Y')
        else:
            print("Coming Soon!!!")
        self.input_dim = self.input_data.shape[1]
        self.output_dim = self.output_data.shape[1]
        self.data_size = self.input_data.shape[0]
        print("Data is Processed")

    def LoadTestData(self, load_path, type_normalize=0):
        if (type_normalize == 0):
            self.input_test = (np.float32(np.loadtxt(load_path + '/Input.txt')) - self.input_mean) / self.input_std
            self.output_test = (np.float32(np.loadtxt(load_path + '/Output.txt')) - self.output_mean) / self.output_std
        else:
            print("Coming Soon!!!")
        self.data_size_test = self.input_test.shape[0]
        print("TestData is Processed")

    def BuildConstantPlaceHolder(self):
        self.nn_batch_size = tf.placeholder(tf.int32, name=self.name_tensor_batch_size)
        self.nn_X = tf.placeholder(tf.float32, [None, self.input_dim], name=self.name_tensor_input)
        self.nn_Y = tf.placeholder(tf.float32, [None, self.output_dim], name=self.name_tensor_output)
        self.nn_keep_prob = tf.placeholder(tf.float32, name=self.name_tensor_keep_prob)
