Local Motion Phases for Learning Multi-Contact Character Movements
============
Sebastian Starke, Yiwei Zhao, Taku Komura, Kazi Zaman

ACM Transactions on Graphics / SIGGRAPH 2020

How to play the demo?
------------
This repository contains the code and demo project for the local motion phases technology. An already compiled demo is available for Windows, supporting both Gamepad and Keyboard inputs, and can be downloaded here:

http://www.starke-consult.de/AI4Animation/SIGGRAPH_2020/Demo_Windows.zip

After launching the program, the menu allows you to select from 3 different demos, including interactive basketball plays, generative control sampling, and a quadruped locomotion controller. You will find details on how to control the character movements with joystick, buttons or keys inside each demo application in the info panel at the top. You can also launch the demo in Unity itself by opening the scene file at Unity -> Assets -> Demo -> Demo.unity.

------------
Interactive Basketball Plays

This demo showcases how local motion phases can enhance asynchronous character movements for basketball plays and different skills such as dribbling, shooting, and sudden ball maneuvers while turning or sprinting.
You may adjust the parameters in the script inspector or inside the code and observe how the movements may change according to code logic that maps the user controls to network inputs.

------------

Generative Control Sampling

This demo produces variations of the given user controls by injecting noise into the latent space of a conditional GAN architecture. The noise seed enables easily switching between different motion apperances of the same skill,
and the noise scale defines the amount of variation by altering the latent space of the network. Note that this value should usually be within the range 0.5 and 1.0, where larger values may cause extrapolated movements.

------------

Quadruped Locomotion Control

This demo shows how the same local phase feature can be used to learn the complex nature of quadruped locomotion modes. This was previously done by using the foot velocity magnitues in the 2018 work, but which did not contain temporal information of the motion and further did not work very well on biped locomotion. Using local phase variables can help generalizing better to both such examples as demonstrated by the demos in this work.

------------

How to reproduce the results?
------------
The complete code that was used for processing, training, and generating the basketball movements is provided in this repository. However, the basketball motion data itself unfortunately can not be shared, so the code serves more like a reference implementation for your own use cases. Instead, a complete walkthrough to reproduce the quadruped example demo is given below, but which is conceptually similar to the basketball example.

------------

Starting with the downloaded Unity project.

#1 Open the Motion Capture Scene, located in Unity -> Assets -> Demo -> Quadruped -> MotionCapture.unity.

#2 Click on the MotionEditor game object in the scene hierarchy window.

#3 Open the Motion Exporter (Header -> AI4Animation -> Exporter -> Motion Exporter). Make sure "Quadruped" pipeline is selected. You may set "Frame Shifts" to 1 and "Frame Buffer" to 30, and have the box for "Write Mirror" checked.

#4 Click the "Export" button, which will generate the training data and save it in the DeepLearning folder.

#5 Navigate to the DeepLearning -> Models -> ExpertModel folder, and save this data into the dataset folder.

#5 Run the main.py file which will start the training.

#6 Wait for a few hours.

#7 You will find the trained binary network weights in the training folder.

#8 Go back to Unity and copy the path to that folder into the inspector field of the ExpertModel component of the character.

#9 The system should run when hitting Play.

------------

Starting with the raw motion capture data.

If you decide to start from the raw motion capture and not use the already processed assets in Unity, you will need to download the quadruped dataset from our 2018 paper and do the following steps:

#1 Import the motion data into Unity by opening the BVH Importer (Header -> AI4Animation -> Importer -> BVH Importer). Define the path where the original .bvh data is saved on your hard disk, and where the Unity assets shall be saved inside the project.

#2 Press "Load Directory" and "Import Motion Data".

#3 Create a new scene, add an empty game object and add the MotionEditor component to it.

#4 Copy the path where the imported motion data assets have be saved and click "Import".

#5 In the "Editor Settings" at the bottom, make sure that "Target Framerate" is set to 30Hz.

#6 Open the MotionProcessor window (Header -> AI4Animation -> Tools -> MotionProcessor), make sure that "Quadruped Pipeline" is selected and click "Process".

#5 Wait for a few hours.

#6 At this point, the raw motion capture data has been automatically processed and is at the same stage as the motion assets provided in this repository. You are ready to continue with the steps above to export the data, train the network and control the character movements.

Additional information
------------

For the character control demos, note that the motion apperances may depend on how the user controls the gamepad joysticks for example. This means, a little practise moving the right joystick for ball dribbling maneuvers will likely improve the quality of those. Similarly, wildly spinning the left joystick for the quadruped locomotion demo may also result in unexpected movements. While this is partially due to missing data examples for such inputs, additional filtering of the user controls may be required in an actual game integration. For the generative control sampling, since some seeds may not give the desired results, a typical workflow would be to save a small set of predefined seed and scale values that are known to be good, and randomly sample from those to dynamically change the character movements during runtime. The scale can further be adjusted dynamically based on how much the user is controlling the character, or how much it should be controlled by the AI. All the pretrained weights used in the demo can further be found in the DeepLearning -> Weights folder. Lastly, the local phase technology is not strictly bound to be used with neural networks, but can also help to enhance other animation systems where clustering similar motions is of importance.

------------
Any questions, feel free to ask. For any issues you might find, please let me know and send me a message to Sebastian.Starke@ed.ac.uk.