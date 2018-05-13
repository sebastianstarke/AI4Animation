AI4Animation
======================================================

Copyright Information
------------
This code implementation is only for research or education purposes, and (especially the learned data) not available for commercial use, redistribution, sublicensing etc. The intellectual property and code implementation belongs to the University of Edinburgh. For scientific use, please reference this repository together with the relevant publications below. In any case, I would ask you to contact me if you intend to seriously use, redistribute or publish anything related to this code or repository.

Description
------------
This project explores the opportunities of deep learning and biologically-inspired optimisation for character animation and control as part of my Ph.D. research at the University of Edinburgh in the School of Informatics, supervised by Taku Komura. The development is done using Unity3D / Tensorflow, and the implementations are made available during my Ph.D. progress.

SIGGRAPH 2018
------------
<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/SIGGRAPH_2018/Teaser.png" width="100%">
Animating characters is a pain, right? Especially those four-legged monsters!<br /><br />
This year, we will be presenting our recent research on quadruped animation and character control at the SIGGRAPH 2018 in Vancouver.
The system can produce natural animations from real motion data using a novel neural network architecture, called Mode-Adaptive Neural Networks.

[![Mode-Adaptive Neural Networks for Quadruped Motion Control](https://img.youtube.com/vi/uFJvRYtjQ4c/0.jpg)](https://www.youtube.com/watch?v=uFJvRYtjQ4c)

SIGGRAPH 2017
------------
This work continues the recent work on PFNN (Phase-Functioned Neural Networks) for character control.

Video: https://www.youtube.com/watch?v=Ul0Gilv5wvY

Paper: http://theorangeduck.com/media/uploads/other_stuff/phasefunction.pdf

A demo in Unity3D using the original weights for terrain-adaptive locomotion is contained in the Assets/Demo/SIGGRAPH_2017/Original folder.
Another demo on flat ground using the Adam character is contained in the Assets/Demo/SIGGRAPH_2017/Adam folder.
In order to run them, you need to download the neural network weights from the link provided in the Link.txt file, extract them into the /NN folder, 
and store the parameters via the custom inspector button.
<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/images/SIGGRAPH_2017/Demo.png" width="100%">

Motion Capture
------------
In progress.