Neural State Machine for Character-Scene Interactions
============================================================

-------------------------------------------------------------
This is a quick walk-through how to play the demo:
-------------------------------------------------------------
#1 Open the Demo Scene (Assets -> Demo -> Demo.scene)
#2 Hit Play
#3 Move around with W,A,S,D (Move), Q,E (Turn), Left-Shift (Sprint)
#4 Move mouse over object and hold key C (Sit), V (Carry) or F (Open) pressed
#5 To sit, keep the C button pressed. To stand up, simply press W to move forward.
#6 To pick up an object, keep the V button pressed. To place it down, release V and
move define the location with the right mouse button (optional).
#7 To open a door, keep the F button pressed.

-------------------------------------------------------------
This is a quick walk-through how to reproduce the results:
-------------------------------------------------------------
#1 Open the Mocap Scene (Assets -> MotionCapture -> Scene.scene)
#2 Open the Motion Exporter (Header -> AI4Animation -> Motion Exporter)
#3 Export the data into TensorFlow / Export folder (create), and then move it into the TensorFlow / data folder
#4 Run the main.py file with TensorFlow 2.0 located in TensorFlow / NSM / Main / main.py
#5 Wait for 1-2 days.
#6 You will find the trained weights in the TensorFlow / trained folder
#7 Put that folder path into the NSM inspector field
#8 The system should run when hitting Play

-------------------------------------------------------------
Additional information
-------------------------------------------------------------
The MoCap data is available in the .asset files, combined
with the location and interaction of objects.

There will be many corner cases where the system can fail
due to the exponential combinatorial amount of possible 
actions / interactions of the character with the environment.

If you have any questions, please contact me at Sebastian.Starke@ed.ac.uk
