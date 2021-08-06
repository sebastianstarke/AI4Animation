Neural State Machine for Character-Scene Interactions
============
Sebastian Starke*, He Zhang*, Taku Komura, Jun Saito (*Joint First Authors)

ACM Transactions on Graphics / SIGGRAPH Asia 2019

How to play the demo?
------------
The demo project is available in the Unity project which can be downloaded from this repository.

#1 Open the Demo Scene (Unity -> Assets -> Demo -> Demo.unity).

#2 Hit the Play button.

#3 Move around with W,A,S,D (Move), Q,E (Turn), Left-Shift (Sprint).

#4 Move mouse over object and hold key C (Sit), V (Carry) or F (Open) pressed.

#5 To sit, keep the C button pressed. To stand up, simply press W to move forward.

#6 To pick up an object, keep the V button pressed. To place it down, release V and move define the location with the right mouse button (optional).

#7 To open a door, keep the F button pressed.

How to reproduce the results?
------------
#1 Open the Mocap Scene (Unity -> Assets -> MotionCapture -> Scene.unity).

#2 Click on the MotionEditor game object in the scene hierarchy window.

#3 Open the Motion Exporter (Header -> AI4Animation -> Motion Exporter).

#4 Export the data into TensorFlow / Export folder (create), and then move it into the TensorFlow / data folder.

#5 Run the main.py file with TensorFlow 2.0 located in TensorFlow / NSM / Main / main.py

#6 Wait for 1-2 days.

#7 You will find the trained binary network weights in the TensorFlow / trained folder.

#8 Go back to Unity and copy the path to that folder into the inspector field of the NSM component of the character.

#9 The system should run when hitting Play.

Additional information
------------
In the demo, there will be many corner cases where the system may fail due to the exponential combinatorial amount of possible actions and interactions of the character with the environment.

The processed motion capture data is available in the .asset files, combined with the location and interaction of objects. If you wish to go from scratch or use the data in your own projects, the raw motion capture files are available here:

http://www.starke-consult.de/AI4Animation/SIGGRAPH_Asia_2019/MotionCapture.zip

------------
Any question, feel free to ask. For any issues you might find, please let me know and send me a message to Sebastian.Starke@ed.ac.uk.