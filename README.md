AI4Animation
======================================================

Description
------------
This project explores the opportunities of deep learning and evolutionary computation for character animation as part of my Ph.D. research at the University of Edinburgh in the School of Informatics, supervised by Taku Komura. It extends the recent work on character control using PFNN (Phase-Functioned Neural Networks: https://www.youtube.com/watch?v=Ul0Gilv5wvY) by generating task-specific procedural animation by learning different motion manifolds. The development is done using Unity3D.

The algorithmic framework is shown below. In addition to the extended PFNN version which utilises multiple phase modules, a memetic evolutionary algorithm for generic inverse kinematics (BioIK: https://github.com/sebastianstarke/BioIK Video: https://www.youtube.com/watch?v=ik45v4WRZKI) is used for animation post-processing.

<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/Cycle.png" width="100%">

Development Status
------------
The code for the PFNN is implemented using MathNet.Numerics, and uses the externally trained weights from Theano to generate the motion of the characters. Those can be imported during edit time and serialised inside Unity. The trajectory estimation module has been implemented which is required for generating the input for the network through user input, as shown below. It provides a smooth transition between past and future states, and rejects paths which would collide with obstacles. The output of the joint positions and velocities along with predicted trajectory information is then fed back to the character to update the posture.

<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/Trajectory.png" width="100%">
<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/Skeleton_1.png" width="100%">
<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/Skeleton_2.png" width="100%">

Usage
------------
Simply import the project into Unity3D, and open the 'Animation' scene. Click on the character, and press the "Load Weights" button in the PFNN tab. Press 'Play', and control your character via W,A,S,D (Move), Q,E (Turn), LeftShift (Run) and LeftCtrl (Crouch).
