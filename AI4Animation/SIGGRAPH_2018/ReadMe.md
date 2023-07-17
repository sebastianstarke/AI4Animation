Mode-Adaptive Neural Networks for Quadruped Motion Control
============
He Zhang*, Sebastian Starke*, Taku Komura, Jun Saito (*Joint First Authors)

ACM Transactions on Graphics / SIGGRAPH 2018

How to play the demo?
------------
This repository contains the code and demo project for the quadruped animation paper above. The compiled demos for Windows / Linux / Mac are available at:

https://www.starke-consult.de/AI4Animation/SIGGRAPH_2018/Demo_Windows.zip

https://www.starke-consult.de/AI4Animation/SIGGRAPH_2018/Demo_Linux.zip

https://www.starke-consult.de/AI4Animation/SIGGRAPH_2018/Demo_Mac.zip

To control the character, use the \<W / A / D> keys for forward and side motion, \<S> for stopping, and \<Q / E> for turning. To control different velocity levels, press \<Left-Shift> (Canter), \<Left-Ctrl> (Trot) or \<Left-Alt> (Walk).
Default locomotion is pace. Jumping, sitting, standing and lying can be performed via \<Space>, \<V>, \<R> or \<F> respectively.

The camera view can be adjusted on the top right. Switching between flat and terrain environment can be done on the top left. Below, you can activate / deactivate different character and data visualisations.

How to reproduce the results?
------------
To generate the parameters yourself by training the network, you will need to download the input and output data into the TensorFlow folder outside of the AI4Animation project, as described in TensorFlow/data/Link.txt.

You can then train the network on this data running the main.py file inside the Tensorflow/MANN folder.

The generated weights are automatically saved inside the Tensorflow/trained folder, which by default are those already imported in the demo in Unity/Assets/Demo/Parameters.asset.

If you wish to start from raw motion capture data, it can be downloaded from this link:

https://www.starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip

You will then need to use the developed motion capture editor tools inside Unity to pre-process the data as you wish, i.e. adding modules and assigning motion labels as mentioned in the paper.

------------
Any questions, feel free to ask. For any issues you might find, please let me know and send me a message to Sebastian.Starke@ed.ac.uk.
