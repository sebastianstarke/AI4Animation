AI4Animation
======================================================

Description
------------
This project explores the opportunities of deep learning and evolutionary computation for character animation as part of my Ph.D. research at the University of Edinburgh in the School of Informatics, supervised by Taku Komura.

It extends the recent research of Daniel Holden, Taku Komura and Jun Saito on character control using PFNN (Phase-Functioned Neural Networks: https://www.youtube.com/watch?v=Ul0Gilv5wvY), and scientifically continues their work for learning task-specific motion manifolds as well as for learning representations for different geometries. The development is done using Unity3D, and the implementation will be made available for character animation research and games development during my Ph.D. progress.

The algorithmic framework is shown below. In addition to the extended PFNN version which utilises multiple phase modules, a memetic evolutionary algorithm for generic inverse kinematics (BioIK: https://github.com/sebastianstarke/BioIK) is used for animation post-processing.

<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/Cycle.png" width="100%">

Development Status
------------
Currently, the code for the PFNN is implemented using MathNet.Numerics, and uses the externally trained weights from Theano to generate the motion of the characters. However, it is not yet mapped to the bones of the character (Kyle) who is currently used inside the project. This is the next task to be done. The trajectory estimation module has been implemented which is required for generating the input for the network through user input, as shown below. It provides a smooth transition between past and future states, and rejects paths which would collide with obstacles.

<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/Trajectory.png" width="100%">
