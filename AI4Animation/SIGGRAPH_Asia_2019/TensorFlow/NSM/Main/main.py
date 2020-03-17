import sys

sys.path.append('../Lib_NSM')
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from Main_EMPNN import MainNN

# Tuning Settings
hidden_size_Gating = 512
hidden_size_Main = 512
keep_prob_components = 0.7
# Index
start_pose = 0
start_goal = 419
start_environment = 575
start_interaction = 2609
start_gating = 4657
dim_gating = 650
# Encoders:
index_encoders = [
    np.arange(start_pose, start_goal),
    np.arange(start_goal, start_environment),
    np.arange(start_environment, start_interaction),
    np.arange(start_interaction, start_gating)
]
dim_encoders = [
    [512, 512],
    [128, 128],
    [512, 512],
    [512, 512]
]
activation_encoders = [
    [tf.nn.elu, tf.nn.elu],
    [tf.nn.elu, tf.nn.elu],
    [tf.nn.elu, tf.nn.elu],
    [tf.nn.elu, tf.nn.elu]
]
assert len(index_encoders) == len(dim_encoders) == len(activation_encoders)
# Path Setting
load_path = '../../data'
save_path = '../../trained'
type_normalization = 0
name_model = "NSM"
expert_components = [
    1,
    10
]
act_components = [
    [tf.nn.elu, tf.nn.elu, tf.nn.softmax],
    [tf.nn.elu, tf.nn.elu, 0]
]
input_components = [
    np.arange(start_gating, start_gating + dim_gating),
    []
]
dim_components = [
    [hidden_size_Gating, hidden_size_Gating],
    [hidden_size_Main, hidden_size_Main]
]

def main():
    sess = tf.Session()
    model = MainNN()
    model.BuildModel(load_path, save_path, type_normalization,
                    expert_components,
                    input_components,
                    dim_components,
                    act_components,

                    index_encoders,
                    dim_encoders,
                    activation_encoders
                    )
    model.Train(sess,
               name_model,
               keep_prob=keep_prob_components,
               flag_save_tf=False, flag_save_bin=True,
               )

if __name__ == '__main__':
    main()
