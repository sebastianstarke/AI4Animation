========== Animation Authoring for Neural Quadruped Controllers ==========

To create a path for the character, simply add the Animation Authoring script
to any gameobject in your scene. Once added you can create controlpoints in the
scene by pressing <Space> (default). To select the corresponding motion action type,
use the sliders in the inspector for changing styles. For each controlpoint you can set up
a motion time (in seconds) that defines a timeinterval to remain at this state.
This is especially required for non-locomotion styles, such as sitting, standing,
eating, and hydrating.
The default time between two points is one second. 

Once you set up the authoring, simply attach it to the Runtime script of the wolf game
character. 

The camera view can be adjusted on the top right. Switching between flat and 
terrain environment can be done on the top left. Below, you can activate / 
deactivate different character and data visualisations.

When running the demo inside the Unity engine you need to import the neural network parameters
generated from the weights in the TensorFlow/Trained_NN_Wolf folder
with the MANN script.

To generate the parameters yourself by training the network, you will need
to run the main.py file in the TensorFlow/MANN folder outside of
the Unity project.
================================================================================