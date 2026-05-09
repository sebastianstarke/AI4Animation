import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from MainNN import MainNN

datapath = './dataset'
file_input = datapath + "/Input.txt" 
file_output = datapath + "/Output.txt" 
norm_input = datapath + "/InputNorm.txt" 
norm_output = datapath + "/OutputNorm.txt"
savepath    = './training'

expert_components    =      [
                            1,
                            8
                            ]

act_components            = [
                            [tf.nn.elu, tf.nn.elu, tf.nn.softmax],
                            [tf.nn.elu, tf.nn.elu, 0]
                            ]

keep_prob_components      = [
                            0.7,
                            0.7
                            ]

input_components          = [
                            [(469 + i) for i in range(104)],
                            [(i) for i in range(469)]
                            ]

output_components         = [
                            ...
                            ]

dim_components =            [
                            [128, 128],
                            [512, 512]
                            ]

def main():
    rng  = np.random.RandomState(23456)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    network = MainNN(
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

                batch_size = 32 , epoch = 70, Te = 10, Tmult = 2, 
                learning_rate_ini = 0.0001, weightDecay_ini = 0.0025
                )
    
    network.build_model()
    network.train()
    
if __name__ =='__main__':
    main()
