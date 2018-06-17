========== Mode-Adaptive Neural Networks for Quadruped Motion Control ==========
He Zhang*, Sebastian Starke*, Taku Komura, Jun Saito (*Joint First Authors)
ACM Transactions on Graphics / SIGGRAPH 2018

This folder contains the demo project for the quadruped animation paper mentioned 
above. The compiled demos for Windows / Linux / Mac are available at:
http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/Demo_Windows.zip
http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/Demo_Linux.zip
http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/Demo_Mac.zip

To control the character, use the <W / A / D> keys for forward and side motion, 
<S> for stopping, and <Q / E> for turning. To control different velocity levels, 
press <Left-Shift> (Canter), <Left-Ctrl> (Trot) or <Left-Alt> (Walk). Default 
locomotion is pace. Jumping, sitting, standing and lying can be performed via 
<Space>, <V>, <R> or <F> respectively.

The camera view can be adjusted on the top right. Switching between flat and 
terrain environment can be done on the top left. Below, you can activate / 
deactivate different character and data visualisations.

When running the demo inside the Unity engine and not using the compiled
runtime demo, you need to download the neural network parameters as described
in Assets/Demo/SIGGRAPH_2018/Link.txt.

To generate the parameters yourself by training the network, you will need
to download the input and output data into the TensorFlow folder outside of
the AI4Animation project, as described in TensorFlow/SIGGRAPH_2018/data/Link.txt.

The raw motion capture data can be downloaded from this link:
http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/MotionCapture.zip
Using the developed motion capture editor tools inside Unity, you can 
pre-process the data as you wish.

Any question, feel free to ask. Any issues you might find, please let me know
and send me a message to Sebastian.Starke@ed.ac.uk.
================================================================================