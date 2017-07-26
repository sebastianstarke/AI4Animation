Description
------------
This project extends the recent research of Daniel Holden, Taku Komura and Jun Saito on character control using Phase-Functioned Neural Networks.

Phase-Functioned Neural Networks for Character Control
======================================================

This project contains the code for Phase-Functioned Neural Networks for 
Character Control as presented at SIGGRAPH 2017, with the project page found
below:

http://theorangeduck.com/page/phase-functioned-neural-networks-character-control

This paper presents a character controller driven by a special kind of network
called a Phase-Functioned Neural Network which learns a mapping from the user
input parameters, the character's previous state, and the surrounding 
environment to the pose of the character at a given frame. The rest of this 
README will assume you have read the paper and understand the basic workings 
and processes involved in this technique.

The code is essentially separated two sub-projects here.

The first project is a set of python scripts written in Numpy/Scipy/Theano 
which preprocess the motion data, generate a database, and train the phase 
functioned neural network to perform the regression. The output of this project
are the network weights saved as simple binary files.

The second project (contained in the subfolder "demo") is a C++ project which
contains a basic runtime that loads the network weights and runs an interactive 
demo which shows the results of the trained network when controlled via a 
game-pad.

Below are details of the steps for reproducing the results from the paper from 
preprocessing and training to runnning the demo.


Installation
------------

Before you do anything else you will first need to install the following python 
packages `numpy, scipy, Pillow, theano` as well as CUDA, cuDNN etc. This 
project was built using Python3 but may work with Python2 given a few minor 
tweaks.


Preprocessing
-------------

The next step is to build the database of terrain patches which are later 
fitted to the motion. For this simply run the following. 

    python generate_patches.py

This will sample thousands of patches from the heightmaps found in 
`data/heightmaps` and store them in a database called `patches.npz`. This
should take a good few minutes so be patient.

Now you can begin the process of preprocessing the animation data - converting 
it into the correct format and fitting the terrain to each walk cycle. For this
you run the following:

    python generate_database.py

This process can take some time - at least a couple of hours. It uses all the 
data found in `data/animations` and once complete will output a database called 
`database.npz`. If you want to change the parameterisation used by the network 
this is probably the place to look - but note that the preprocessing for this 
work is quite fiddily and complicated so you must be careful when you edit this 
script not to introduce any bugs.


Training
--------

Once you've generated `database.npz` you can begin training. For this simply 
run the following:

    python train_pfnn.py

Assuming you've installed `theano`, `CUDA`, and everything else successfully 
this should start training the neural network. This requires quite a lot of RAM
as well as VRAM. If you get any kind of memory error you can perhaps try using 
less data by subsampling the database or even taking some of the clips out of 
the preprocessing stage by removing them from `generate_database.py`.

During the training process the weights of the network will be saved at each
epoch to the location `demo/network/pfnn` so don't worry about stopping the 
training early. It is possible to achieve decent results in just an hour or so
of training, but for the very best results you may need to wait overnight. For
this reason it might be making a backup of the pre-trained demo weights before
beginning training.


Runtime
-------

With the network weights generated you're now ready to try the runtime. For 
instructions for this please look inside the `demo` folder.


