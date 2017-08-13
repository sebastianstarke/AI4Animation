AI4Animation
======================================================

Description
------------
This project explores the opportunities of deep learning and evolutionary computation for character animation as part of my Ph.D. research at the University of Edinburgh in the School of Informatics, supervised by Taku Komura.

It extends the recent research of Daniel Holden, Taku Komura and Jun Saito on character control using PFNN (Phase-Functioned Neural Networks: https://www.youtube.com/watch?v=Ul0Gilv5wvY), and scientifically continues their work for learning task-specific motion manifolds using multiple phases modules, as well as for learning representations for different geometries. The development is done using Unity3D, and the implementation will be made available for character animation research and games development inside this engine.

The algorithmic framework is shown below. In addition to the extended Phase-Functioned Neural Networks, a memetic evolutionary algorithm for generic inverse kinematics (BioIK: https://github.com/sebastianstarke/BioIK) is used for animation post-processing.

<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/Cycle.png" width="100%">

Development Status
------------
Currently, the code for the PFNN is implemented using MathNet.Numerics, and uses the externally trained weights from Theano to generate the motion of the characters. However, it is not yet mapped to the bones of the character (Kyle) which is currently used inside the project. So far, the trajectory generation
