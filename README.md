AI4Animation
======================================================

Description
------------
This project explores the opportunities of deep learning and evolutionary computation for character animation as part of my Ph.D. research at the University of Edinburgh in the School of Informatics, supervised by Taku Komura.

It extends the recent work of Daniel Holden, Taku Komura and Jun Saito on character control using PFNN (Phase-Functioned Neural Networks: https://www.youtube.com/watch?v=Ul0Gilv5wvY) for learning task-specific motion manifolds as well as for learning representations for different geometries. The development is done using Unity3D, and the implementation will be made available for character animation research and games development during my Ph.D. progress.

The algorithmic framework is shown below. In addition to the extended PFNN version which utilises multiple phase modules, a memetic evolutionary algorithm for generic inverse kinematics (BioIK: https://github.com/sebastianstarke/BioIK Video: https://www.youtube.com/watch?v=ik45v4WRZKI) is used for animation post-processing.

<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/Cycle.png" width="100%">

Development Status
------------
The only required script component that is needed for the animation is called 'BioAnimation'. The code for the PFNN is implemented using MathNet.Numerics, and uses the externally trained weights from Theano/TensorFlow to generate the character motion. The weights are supposed to be imported during edit time and are then serialised inside Unity. The trajectory estimation module handles the user input to control the character movements, and rejects paths which would collide with obstacles. The output of the joint positions, velocities, corrections etc. is then fed back to the character to update the posture.

<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/Demo_1.png" width="100%">
<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/Demo_2.png" width="100%">

Demo
------------
To run the demo, simply start Unity and open the provided scene in the Assets folder. You then need to press the "Load Parameters" button in the attached 'BioAnimation' component, which can take a few seconds. When hitting play, the character can be controlled via W-A-S-D (move), Q-E (turn), LeftShift (run) and LeftCtrl (crouch). Jumping is handled automatically given the rise of the terrain.

Data Preprocessing
------------
A 'BVHViewer' editor window has been developed with the aim to do animation preprocessing of BVH files to generate the training data for the deep learning. This enables defining the phase function, style function, generating trajectories as well as all other relevant data right inside Unity.
<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/BVH.png" width="100%">
